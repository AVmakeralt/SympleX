# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import numpy as np

array = np.array
float32 = np.float32



# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_17 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['tpu_custom_call'],
    serialized_date=datetime.date(2026, 2, 18),
    inputs=(),
    expected_outputs=(array(1., dtype=float32),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<f32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<true> : tensor<i1> loc(#loc19)
    %c_0 = stablehlo.constant dense<0> : tensor<i32> loc(#loc20)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc)
    %c_2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<1024xi32> loc(#loc21)
    %1 = stablehlo.reshape %0 : (tensor<1024xi32>) -> tensor<8x128xi32> loc(#loc22)
    %2 = call @remainder(%1, %c_2) : (tensor<8x128xi32>, tensor<i32>) -> tensor<8x128xi32> loc(#loc7)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<8x128xi32> loc(#loc23)
    %4 = stablehlo.compare  EQ, %2, %3,  SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1> loc(#loc23)
    %5 = stablehlo.convert %4 : (tensor<8x128xi1>) -> tensor<8x128xi32> loc(#loc20)
    %6 = stablehlo.convert %4 : (tensor<8x128xi1>) -> tensor<8x128xi32> loc(#loc20)
    %7 = stablehlo.custom_call @tpu_custom_call(%5, %6) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSZ29vZ2xlMy10cnVuawABJQcBAwUBAwcDEwkLDQ8RExUXGQORcREBbw8HEw8PCxcLExMLDwsLDxsLDwszCwsLC4ULCwsPDx8LDw8XExMLDwsPCw8LCw8LFyMLDwtTCxMFA2EBERsHGyMPDwsbAloEHVlbHwMDCzkdGTsdGUMFGxcPegIjBR0DAws9AwM/QQUfEQsBBSEFIx1LTQMFISMVJQUlEQshBScDCykrLS8zFzUXFTcFKQEBBSsND2FmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAUtBS8FMREDAR0bDSUBCQAAAAAFMxELBR0bRRcPegI3AwMLSSUFA/8FNR1PDQU3HVNVBTkdVw0FOwU9HV1fBT8XD3oCEQMHY2VnaWttBUERDQAFQyMJCSEBAAAAAQAAAAIAAAAAAAAABUUjCQEBI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AJwUhAgQJAycFIQIEDRdvBSECBAkxAQICAQIEAQkFBwcHBwEEtgIFAREDHwcDAQUJEQMnBwMtVwcHAwcDBwMDAwcFAwMDAwcFAwMFBgcDAQcBBwkDAwcRAwEHBwcTAwUFCw0DAwkFAwMDAwkFAwMFBgkDAQcDERMDAwkRAwEHBwkTAwUFFRcDAx1HAwULBh0DBQUZGw0GUQMFBQ8dAwMBBQMDAwMBBQMDBQYBAwEHBSEjDwYBAwEDHwMDAREDAQcHARMDBQUlKREFAWEJJwUhIxMAAwYDAQUBAN4GRxEpCQsNBwkJCxULIyEdKQ8tCQsThQ0ZIxkVFxUXGR8PCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAYXJpdGguY21waQBmdW5jLmZ1bmMAYXJpdGgueG9yaQBhcml0aC5vcmkAYXJpdGguZXh0dWkAdHB1LnZlY3Rvcl9zdG9yZQBmdW5jLnJldHVybgB2YWx1ZQB0aGlyZF9wYXJ0eS9weS9qYXgvdGVzdHMvcGFsbGFzL2V4cG9ydF9iYWNrX2NvbXBhdF9wYWxsYXNfdGVzdC5weQBzeW1fbmFtZQBnZXQ6AGdldABzdGFibGVfbW9zYWljLnZlcnNpb24Aa2VybmVsAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHByZWRpY2F0ZQBub3Q6AG5vdABvcjoAb3IAc3dhcDoAc3dhcABhZGQAb3BlcmFuZFNlZ21lbnRTaXplcwBzdHJpZGVzAA==\22, \22serialization_format\22: 1, \22needs_layout_passes\22: true}}", kernel_name = "kernel", mhlo.frontend_attributes = {kernel_metadata = "{}"}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi32> loc(#loc24)
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<8x128xi32> loc(#loc20)
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1> loc(#loc20)
    %10 = stablehlo.convert %9 : tensor<8x128xi1> loc(#loc20)
    %11 = stablehlo.reduce(%10 init: %c) applies stablehlo.and across dimensions = [0, 1] : (tensor<8x128xi1>, tensor<i1>) -> tensor<i1> loc(#loc19)
    %12 = stablehlo.convert %11 : (tensor<i1>) -> tensor<f32> loc(#loc25)
    return %12 : tensor<f32> loc(#loc)
  } loc(#loc)
  func.func private @remainder(%arg0: tensor<8x128xi32> loc(unknown), %arg1: tensor<i32> loc(unknown)) -> tensor<8x128xi32> {
    %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc26)
    %c_0 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0 = stablehlo.convert %arg1 : tensor<i32> loc(#loc9)
    %1 = stablehlo.compare  EQ, %0, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc10)
    %2 = call @_where(%1, %c, %0) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32> loc(#loc11)
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<8x128xi32> loc(#loc12)
    %4 = stablehlo.remainder %arg0, %3 : tensor<8x128xi32> loc(#loc12)
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<8x128xi32> loc(#loc13)
    %6 = stablehlo.compare  NE, %4, %5,  SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1> loc(#loc13)
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<8x128xi32> loc(#loc14)
    %8 = stablehlo.compare  LT, %4, %7,  SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1> loc(#loc14)
    %9 = stablehlo.compare  LT, %2, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<8x128xi1> loc(#loc13)
    %11 = stablehlo.compare  NE, %8, %10,  UNSIGNED : (tensor<8x128xi1>, tensor<8x128xi1>) -> tensor<8x128xi1> loc(#loc13)
    %12 = stablehlo.and %11, %6 : tensor<8x128xi1> loc(#loc15)
    %13 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<8x128xi32> loc(#loc16)
    %14 = stablehlo.add %4, %13 : tensor<8x128xi32> loc(#loc16)
    %15 = stablehlo.select %12, %14, %4 : tensor<8x128xi1>, tensor<8x128xi32> loc(#loc17)
    return %15 : tensor<8x128xi32> loc(#loc26)
  } loc(#loc26)
  func.func private @_where(%arg0: tensor<i1> loc(unknown), %arg1: tensor<i32> loc(unknown), %arg2: tensor<i32> loc(unknown)) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32> loc(#loc17)
    return %0 : tensor<i32> loc(#loc27)
  } loc(#loc27)
} loc(#loc)
#loc1 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":160:13)
#loc2 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":159:10)
#loc3 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":153:10)
#loc4 = loc("jit(func)"(#loc1))
#loc5 = loc("jit(func)"(#loc2))
#loc6 = loc("jit(func)"(#loc3))
#loc7 = loc("jit(func)/jit(remainder)"(#loc3))
#loc8 = loc("jit(func)/jit"(#loc3))
#loc9 = loc("convert_element_type"(#loc3))
#loc10 = loc("eq"(#loc3))
#loc11 = loc("jit(_where)"(#loc3))
#loc12 = loc("rem"(#loc3))
#loc13 = loc("ne"(#loc3))
#loc14 = loc("lt"(#loc3))
#loc15 = loc("and"(#loc3))
#loc16 = loc("add"(#loc3))
#loc17 = loc("select_n"(#loc3))
#loc18 = loc("jit"(#loc3))
#loc19 = loc("reduce_and"(#loc4))
#loc20 = loc(""(#loc5))
#loc21 = loc("iota"(#loc6))
#loc22 = loc("reshape"(#loc6))
#loc23 = loc("eq"(#loc6))
#loc24 = loc("pallas_call"(#loc5))
#loc25 = loc("convert_element_type"(#loc4))
#loc26 = loc("jit:"(#loc8))
#loc27 = loc("jit:"(#loc18))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.13.4\x00\x011\x05\x01\x05!\x01\x03\x0b\x03\x1f\x0f\x13\x17\x1b\x1f#\'+/37;?CG\x03\x02\x02\xc7\'\x01w\x17\x07\x0f\x0f\x0f\x0f\x0b\x0f\x0b\x0f\x0f\x0b\x0f\x0f\x0b\x0b\x0f\x0f\x0b\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0f\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x17\x0b\x17\x0f\x0b\x0f\x0b\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x0f\x03Q\x0f\x0b\x0b\x0b\x0bO\x0f\x0b\x0b\x0b\x1f\x1f\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x13\x0b\x17\x0b\x0b\x13\x1f\x0f\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0fO\x01\x05\x0b\x0f\x03#\x0f\x1b\x0f\x1b\x07\x07\x0f\x07\x13\x07\x1b\x1f\x13\x17\x13\x07\x13\x02\xde\x05\x17\rf\x02\x15\x1f\x1d[)\x1dO\x01\x1d%\'\x1d\x17;\x05%\x1dQ\x01\x05\'\x1d\x11\x01\x11\x03\x05\x05)\x1d\x17?\x1dC\x01\x05+\x05-\x1dM\x01\x1dW\x01\x05/\x1d\x11Y\x1d\x11]\x1d\x1f\x13\x1d%\x03\x03\x07135\x157\x15\x051\x11\x01\x00\x053\x055\x057\x1d=\x01\x059\x1dA\x01\x05;\x05=\x1d\x1d\x01\x1d\x1f\x01\x1dK\x01\x05?\x05A\x05C\x05E\x1dU\x01\x05G\x05I\x17\r\x82\x02\x1b\x05K\x17\r~\x02\x15\x1da\x13\x05M\x1de\x13\x05O\x1di\x01\x05Q\x03\x05m\xb1o\xb3\x05S\x05U\x1ds)\x05W\x1d\x1d\'\x1f\x1d\x01\r\x01\t\x07\x03\x01\x07\x03\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03y\x1dY\x1d[\x1d]\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x00\x00\x00\x00\x07\x01\x07\x0b#\x15\x03\x03\x97\r\x03\x99\x9b\x1d_\x1da\x1dc\x1de\x03\x05yy#\x19\x03\x07yyy#\x1b\t\t\x1f\t\x03\xff\x1f\x05\t\x02\x00\x00\x00\x13\x0f\x01\x1dg\r\x03\xb5\xb7\x1di\x1dk\x0b\x03\x1dm\x1do\x05\x01\x03\x05\x81\x81\x03\x03\x81\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x01\r)\x05!\x02\x04\r)\x01\x13)\x05!\x02\x04\x13\x1b\x1d)\x01\x17\x01\x11\x01\x03\x11\t\x11\x05\x07\x05\x03\x07\x11\x07\t\x05\x05\x03\x05)\x03\x01\x0f)\x03\x02 \r)\x03\t#\x13)\x03\t\x0f\x04\xe6\x05\x05\x01Q\x03/\x01\x07\x04\xbe\x05\x03\x01\r\rP\x03\x03\x07\x04\x82\x02\x03#I\x07B\t\x05\x03\t\x07B\x05\x07\x03\x05\x07B\x03\t\x03\x05\x07B\x03\x0b\x03\x05\x19B_\r\x03\x1f\x1b\x06c\x03\x07\x03\t\x11Fg\x0f\x03\x07\x05\x0b\x07\x05F+\x11\x03\x07\x03\x05\x03F+\x13\x03\x0b\x05\r\x0f\t\x06\x05\x03\x07\x03\x11\t\x06\x05\x03\x07\x03\x11\x1dGqk\x15\x03\x07\x05\x13\x15\x05F\x05\x11\x03\x07\x03\x03\x03F\x05\x17\x03\x0b\x05\x17\x19\t\x06\x05\x03\x0b\x03\x1b\x1fV\t\x19\x03\t\x05\x1d\x01\x07\x04-\x03\x07\x0b\x05\x13-\x13-\x00\x13\x06\t\x03\t\x05\x01\x03\x0b\x04\t\x03\x05\t\x06u\x03\x11\x03\x1f\x0b\x04\x03\x03!\rP\x0b\x1b\x07\x04~\x02\x03)O\x05\r\t\x00\x07B\x0b\t\x03\x05\x07B\x03\x07\x03\x05\t\x06E\x03\x05\x03\x03\x03FG\x13\x03\t\x05\t\x07\x11FI\x1d\x03\x05\x07\x0b\x05\t\x05F!\x11\x03\x07\x03\r\x15\x06!\x03\x07\x05\x01\x0f\x05F\x07\x11\x03\x07\x03\x07\x03F\x07\x17\x03\x0b\x05\x11\x13\x05F\x0f\x11\x03\x07\x03\x07\x03F\x0f\x1f\x03\x0b\x05\x11\x17\x03F\x0f\x1f\x03\t\x05\r\x07\x05F\x07\x11\x03\x0b\x03\x1b\x03F\x07!\x03\x0b\x05\x19\x1d\x13\x06S\x03\x0b\x05\x1f\x15\x05F#\x11\x03\x07\x03\r\x17\x06#\x03\x07\x05\x11#\x0f\x06\x1b\x03\x07\x07!%\x11\x0b\x04\x0b\x03\'\rP\x19#\x07\x04-\x03\t\x0b\x07\x11\t\t\x00\x0f\x06\x1b\x03\x05\x07\x01\x03\x05\x0b\x04\x19\x03\x07\x06\x03\x01\x05\x01\x00Z!q!\xc6\x16\x07!\x0f\x0f\x0b\x0f!\x0f\x11\x15\x193\x193\x11\x0b\x03\t\t\x07\x07\t\x19\x13\t\x1d\x13%)9\x17\x07+\x0b\x15\x85\x15\x1f\x17\x11\x0f\x1b\x0f\x11\x15\x11\x15\x17\x19)\x17\x0f\x0b\x11builtin\x00vhlo\x00module\x00compare_v1\x00broadcast_in_dim_v1\x00constant_v1\x00convert_v1\x00return_v1\x00func_v1\x00select_v1\x00call_v1\x00and_v1\x00remainder_v1\x00add_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00reduce_v1\x00third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py\x00jit(func)\x00jit:\x00convert_element_type\x00eq\x00reduce_and\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit\x00jit\x00select_n\x00jit(_where)\x00rem\x00ne\x00lt\x00and\x00add\x00\x00iota\x00reshape\x00jit(func)/jit(remainder)\x00kernel_name\x00mhlo.frontend_attributes\x00pallas_call\x00remainder\x00private\x00_where\x00jax.result_info\x00result\x00main\x00public\x00kernel\x00kernel_metadata\x00{}\x00{"custom_call_config": {"body": "TUzvUgFNTElSZ29vZ2xlMy10cnVuawABJQcBAwUBAwcDEwkLDQ8RExUXGQORcREBbw8HEw8PCxcLExMLDwsLDxsLDwszCwsLC4ULCwsPDx8LDw8XExMLDwsPCw8LCw8LFyMLDwtTCxMFA2EBERsHGyMPDwsbAloEHVlbHwMDCzkdGTsdGUMFGxcPegIjBR0DAws9AwM/QQUfEQsBBSEFIx1LTQMFISMVJQUlEQshBScDCykrLS8zFzUXFTcFKQEBBSsND2FmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAUtBS8FMREDAR0bDSUBCQAAAAAFMxELBR0bRRcPegI3AwMLSSUFA/8FNR1PDQU3HVNVBTkdVw0FOwU9HV1fBT8XD3oCEQMHY2VnaWttBUERDQAFQyMJCSEBAAAAAQAAAAIAAAAAAAAABUUjCQEBI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AJwUhAgQJAycFIQIEDRdvBSECBAkxAQICAQIEAQkFBwcHBwEEtgIFAREDHwcDAQUJEQMnBwMtVwcHAwcDBwMDAwcFAwMDAwcFAwMFBgcDAQcBBwkDAwcRAwEHBwcTAwUFCw0DAwkFAwMDAwkFAwMFBgkDAQcDERMDAwkRAwEHBwkTAwUFFRcDAx1HAwULBh0DBQUZGw0GUQMFBQ8dAwMBBQMDAwMBBQMDBQYBAwEHBSEjDwYBAwEDHwMDAREDAQcHARMDBQUlKREFAWEJJwUhIxMAAwYDAQUBAN4GRxEpCQsNBwkJCxULIyEdKQ8tCQsThQ0ZIxkVFxUXGR8PCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAYXJpdGguY21waQBmdW5jLmZ1bmMAYXJpdGgueG9yaQBhcml0aC5vcmkAYXJpdGguZXh0dWkAdHB1LnZlY3Rvcl9zdG9yZQBmdW5jLnJldHVybgB2YWx1ZQB0aGlyZF9wYXJ0eS9weS9qYXgvdGVzdHMvcGFsbGFzL2V4cG9ydF9iYWNrX2NvbXBhdF9wYWxsYXNfdGVzdC5weQBzeW1fbmFtZQBnZXQ6AGdldABzdGFibGVfbW9zYWljLnZlcnNpb24Aa2VybmVsAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHByZWRpY2F0ZQBub3Q6AG5vdABvcjoAb3IAc3dhcDoAc3dhcABhZGQAb3BlcmFuZFNlZ21lbnRTaXplcwBzdHJpZGVzAA==", "serialization_format": 1, "needs_layout_passes": true}}\x00tpu_custom_call\x00\x08{%\x05s\x01\x0b}\x93\x95\x9d\x9f\x03\xab\x03\x8d\x03\x8b\x03\xad\x03\xaf\x03\x85\x03w\x05{\x8f\x11\xb9\xbb\xbd}\xbf\xc1}\xc3\x05{\x7f\x03\xc5\x0b\xa1\xa3\x83\x85\x87\x03\x89\x05{\x91\x05\xa9\x7f\x0b\xa5\xa7\x83\x89\x87',
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
