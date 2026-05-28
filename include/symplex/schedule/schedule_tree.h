// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <variant>
#include <functional>
#include <cassert>

namespace symplex::schedule {

/// ScheduleTree: a tree structure representing the polyhedral schedule.
/// Each node represents a scheduling decision (loop, filter, extension, etc.)
/// and the leaves represent individual statements.
///
/// The tree directly corresponds to the ISL schedule tree representation:
///   - Domain node: sets the iteration domain for a subtree
///   - Filter node: selects a subset of the domain
///   - Band node: represents a set of affine loops (tile bands)
///   - Leaf node: represents a statement execution

enum class ScheduleNodeType {
    DOMAIN,
    FILTER,
    BAND,
    SEQUENCE,
    SET,
    LEAF,
    CONTEXT,
    EXTENSION,
};

inline std::string schedule_node_type_to_string(ScheduleNodeType t) {
    switch (t) {
        case ScheduleNodeType::DOMAIN:    return "Domain";
        case ScheduleNodeType::FILTER:    return "Filter";
        case ScheduleNodeType::BAND:      return "Band";
        case ScheduleNodeType::SEQUENCE:  return "Sequence";
        case ScheduleNodeType::SET:       return "Set";
        case ScheduleNodeType::LEAF:      return "Leaf";
        case ScheduleNodeType::CONTEXT:   return "Context";
        case ScheduleNodeType::EXTENSION: return "Extension";
    }
    return "Unknown";
}

/// BandMember: describes one loop dimension in a band node.
struct BandMember {
    std::vector<int64_t> coefficients;  // Affine coefficients for each iterator
    int64_t               constant;       // Constant offset
    bool                  parallel;       // Can this loop be parallelized?
    bool                  coincidence;    // Does this loop carry dependencies?

    BandMember() : constant(0), parallel(false), coincidence(false) {}

    BandMember(size_t ndim, int64_t off = 0)
        : coefficients(ndim, 0), constant(off), parallel(false), coincidence(false) {}
};

/// BandNodeData: data for a BAND node.
struct BandNodeData {
    std::vector<BandMember> members;  // One per loop dimension in the band
    bool permutable;                   // Can the loops in this band be arbitrarily reordered?

    BandNodeData() : permutable(false) {}
};

/// FilterNodeData: data for a FILTER node.
struct FilterNodeData {
    std::string statement_name;    // Which statement this filters
    // In a full implementation, this would contain the actual polyhedral set
    std::vector<int64_t> lower_bounds;
    std::vector<int64_t> upper_bounds;
};

class ScheduleTree;
using ScheduleTreePtr = std::shared_ptr<ScheduleTree>;

class ScheduleTree : public std::enable_shared_from_this<ScheduleTree> {
public:
    // ── Construction ────────────────────────────────────────────────

    static ScheduleTreePtr create(ScheduleNodeType type) {
        return ScheduleTreePtr(new ScheduleTree(type));
    }

    // ── Node type ───────────────────────────────────────────────────

    [[nodiscard]] ScheduleNodeType type() const { return type_; }
    void set_type(ScheduleNodeType t) { type_ = t; }

    // ── Children ────────────────────────────────────────────────────

    [[nodiscard]] const std::vector<ScheduleTreePtr>& children() const { return children_; }

    ScheduleTreePtr add_child(ScheduleNodeType type) {
        auto child = ScheduleTree::create(type);
        child->parent_ = this;
        children_.push_back(child);
        return child;
    }

    void add_child(ScheduleTreePtr child) {
        child->parent_ = this;
        children_.push_back(std::move(child));
    }

    [[nodiscard]] ScheduleTree* parent() const { return parent_; }

    // ── Band-specific data ──────────────────────────────────────────

    [[nodiscard]] BandNodeData& band_data() { return band_data_; }
    [[nodiscard]] const BandNodeData& band_data() const { return band_data_; }

    // ── Filter-specific data ────────────────────────────────────────

    [[nodiscard]] FilterNodeData& filter_data() { return filter_data_; }
    [[nodiscard]] const FilterNodeData& filter_data() const { return filter_data_; }

    // ── Domain name ─────────────────────────────────────────────────

    [[nodiscard]] const std::string& domain_name() const { return domain_name_; }
    void set_domain_name(std::string n) { domain_name_ = std::move(n); }

    // ── Tree traversal ──────────────────────────────────────────────

    /// Depth-first traversal.
    void dfs(std::function<void(ScheduleTreePtr)> visitor) {
        visitor(shared_from_this());
        for (auto& child : children_) {
            child->dfs(visitor);
        }
    }

    /// Const depth-first traversal.
    void dfs(std::function<void(const ScheduleTree&)> visitor) const {
        visitor(*this);
        for (const auto& child : children_) {
            child->dfs(visitor);
        }
    }

    /// Find all nodes of a given type.
    [[nodiscard]] std::vector<ScheduleTreePtr> find_nodes(ScheduleNodeType t) {
        std::vector<ScheduleTreePtr> result;
        dfs([&](ScheduleTreePtr node) {
            if (node->type() == t) result.push_back(node);
        });
        return result;
    }

    /// Get the band nodes (the tileable loops).
    [[nodiscard]] std::vector<ScheduleTreePtr> band_nodes() {
        return find_nodes(ScheduleNodeType::BAND);
    }

    /// Get leaf nodes.
    [[nodiscard]] std::vector<ScheduleTreePtr> leaf_nodes() {
        return find_nodes(ScheduleNodeType::LEAF);
    }

    // ── Tree manipulation ───────────────────────────────────────────

    /// Insert a band node between this node and its children.
    /// The band captures the outermost `n` dimensions.
    ScheduleTreePtr band_insert(size_t n_dims) {
        auto band = ScheduleTree::create(ScheduleNodeType::BAND);
        band->band_data_.members.resize(n_dims);
        band->band_data_.permutable = true;

        // Move all children of this node to be children of the band
        band->children_ = std::move(children_);
        for (auto& child : band->children_) {
            child->parent_ = band.get();
        }
        children_.clear();
        add_child(band);
        return band;
    }

    /// Tile a band node: split each member into an outer and inner pair.
    /// Returns the new inner band node.
    ScheduleTreePtr tile_band(const std::vector<int64_t>& tile_sizes) {
        assert(type_ == ScheduleNodeType::BAND);
        size_t n = band_data_.members.size();
        assert(tile_sizes.size() == n);

        // Create an inner band with the same number of members
        auto inner_band = ScheduleTree::create(ScheduleNodeType::BAND);
        inner_band->band_data_.members = band_data_.members;
        inner_band->band_data_.permutable = band_data_.permutable;

        // Mark the outer band members as tile loops
        for (size_t i = 0; i < n; ++i) {
            // The outer loop steps by tile_sizes[i]
            // The inner loop runs 0..tile_sizes[i]-1
            // This is encoded by modifying the coefficients
            // Original: schedule(i) = coeff * i + const
            // Tiled:    schedule(i) = coeff * tile_size * outer_i + coeff * inner_i + const
            if (band_data_.members[i].coefficients.empty()) {
                band_data_.members[i].coefficients.resize(n, 0);
            }
            band_data_.members[i].coefficients[0] = tile_sizes[i];  // Simplified
            inner_band->band_data().members[i].constant = 0;
        }

        // Move children to inner band
        inner_band->children_ = std::move(children_);
        for (auto& child : inner_band->children_) {
            child->parent_ = inner_band.get();
        }
        children_.clear();
        add_child(inner_band);

        return inner_band;
    }

    /// Mark a band member as parallel.
    void mark_parallel(size_t member_idx) {
        assert(type_ == ScheduleNodeType::BAND);
        assert(member_idx < band_data_.members.size());
        band_data_.members[member_idx].parallel = true;
        band_data_.members[member_idx].coincidence = false;
    }

    /// Mark a band member as sequential (carries dependencies).
    void mark_sequential(size_t member_idx) {
        assert(type_ == ScheduleNodeType::BAND);
        assert(member_idx < band_data_.members.size());
        band_data_.members[member_idx].parallel = false;
        band_data_.members[member_idx].coincidence = true;
    }

    // ── String representation ───────────────────────────────────────

    [[nodiscard]] std::string to_string(int indent = 0) const {
        std::ostringstream oss;
        std::string pad(indent, ' ');
        oss << pad << schedule_node_type_to_string(type_);

        if (type_ == ScheduleNodeType::BAND) {
            oss << " [n_loops=" << band_data_.members.size()
                << ", permutable=" << band_data_.permutable << "]";
            for (size_t i = 0; i < band_data_.members.size(); ++i) {
                oss << "\n" << pad << "  loop" << i
                    << (band_data_.members[i].parallel ? " [parallel]" : " [sequential]");
            }
        } else if (type_ == ScheduleNodeType::FILTER) {
            oss << " [stmt=" << filter_data_.statement_name << "]";
        } else if (type_ == ScheduleNodeType::DOMAIN) {
            oss << " [name=" << domain_name_ << "]";
        }

        for (const auto& child : children_) {
            oss << "\n" << child->to_string(indent + 2);
        }
        return oss.str();
    }

private:
    ScheduleTree(ScheduleNodeType type)
        : type_(type) {}

    ScheduleNodeType type_;
    ScheduleTree* parent_ = nullptr;
    std::vector<ScheduleTreePtr> children_;

    // Node-specific data
    BandNodeData   band_data_;
    FilterNodeData filter_data_;
    std::string    domain_name_;
};

// ── Tree utility functions (no IterationSpace dependency) ─────────────────────

/// Compute the depth of the schedule tree.
int tree_depth(const ScheduleTree& tree);

/// Count the total number of nodes in the schedule tree.
size_t tree_size(const ScheduleTree& tree);

/// Find the first band node in the tree (BFS order).
ScheduleTreePtr find_first_band(const ScheduleTreePtr& tree);

/// Collect all leaf statement names from the schedule tree.
std::vector<std::string> collect_statement_names(const ScheduleTreePtr& tree);

} // namespace symplex::schedule
