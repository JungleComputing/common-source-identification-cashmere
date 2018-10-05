/*
 * Copyright 2018 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nl.junglecomputing.common_source_identification.main_mem_cache;

class CorrelationMatrix implements java.io.Serializable {

    private static final long serialVersionUID = 1L;

    double[][] coefficients;

    CorrelationMatrix(int n) {
        coefficients = new double[n][n];
    }

    void add(Correlation correlation) {
        coefficients[correlation.i][correlation.j] = correlation.coefficient;
        coefficients[correlation.j][correlation.i] = correlation.coefficient;
    }
}