/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector.h"

#include <assert.h>

#include <iomanip>

#include "matrix.h"
#include "utils.h"

namespace fasttext {

Vector::Vector(int64_t m) {
  m_ = m; // 向量的大小
  data_ = new real[m]; // 向量的值,real 就是 float 类型
}

Vector::~Vector() {
  delete[] data_; // 删除向量
}

int64_t Vector::size() const {
  return m_; // 获取向量的大小
}
/*
* 初始化向量的值为 0.0
*/
void Vector::zero() {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
  }
}
/*
* vector中的每一个元素都乘以 a
*/
void Vector::mul(real a) {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] *= a;
  }
}
/*
* vec j 行的值 加上 Matrix i 行的值
*/
void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += A.data_[i * A.n_ + j];
  }
}
/*
* vec j 行的值 加上 Matrix i 行的值, Matrix i 行的权重 是 a 
*/
void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += a * A.data_[i * A.n_ + j];
  }
}
/*
* 乘法：Matrix i 行 的值 乘以 Vector 的值，结果加上 Vector data_ 对应元素的值
*/
void Vector::mul(const Matrix& A, const Vector& vec) {
  assert(A.m_ == m_);
  assert(A.n_ == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
    for (int64_t j = 0; j < A.n_; j++) {
      data_[i] += A.data_[i * A.n_ + j] * vec.data_[j];
    }
  }
}
/*
* data_ 的最大值，返回最大值的下标
*/
int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < m_; i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}
/*
* 获取 i 位置的值
*/
real& Vector::operator[](int64_t i) {
  return data_[i];
}

const real& Vector::operator[](int64_t i) const {
  return data_[i];
}
/*
* 输出
*/
std::ostream& operator<<(std::ostream& os, const Vector& v)
{
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.m_; j++) {
    os << v.data_[j] << ' ';
  }
  return os;
}

}
