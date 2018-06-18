/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <assert.h>

#include <algorithm>

#include "utils.h"

namespace fasttext {

Model::Model(std::shared_ptr<Matrix> wi,
             std::shared_ptr<Matrix> wo,
             std::shared_ptr<Args> args,
             int32_t seed)
  : hidden_(args->dim), output_(wo->m_), grad_(args->dim), rng(seed)
{
  wi_ = wi; // n 行 m列矩阵
  wo_ = wo; // n 行 m列矩阵
  args_ = args;
  isz_ = wi->m_; // m 列的 一维向量
  osz_ = wo->m_; // m 列的 一维向量
  hsz_ = args->dim; // 词向量的维度
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  initSigmoid(); // sigmoid 函数初始化
  initLog();
}

Model::~Model() {
  delete[] t_sigmoid;
  delete[] t_log;
}
/*
* 二元逻辑回归
*/
real Model::binaryLogistic(int32_t target, bool label, real lr) {
  real score = sigmoid(wo_->dotRow(hidden_, target)); //hidden_ 乘以 target 就是目标值的分数
  real alpha = lr * (real(label) - score);            // lr 是学习率，真实值 - 预测值， 这里是 计算梯度
  grad_.addRow(*wo_, target, alpha);                  //  更新梯度
  wo_->addRow(hidden_, target, alpha);                // 更新 参数
  if (label) {                                        // 获取 sigmoid 分数
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}
/*
* 根据正负样本，训练时候参数不一样
*/
real Model::negativeSampling(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr);
    } else {
      loss += binaryLogistic(getNegative(target), false, lr);
    }
  }
  return loss;
}
/*
* 。在霍夫曼树中，隐藏层到输出层的softmax映射不是一下子完成的，而是沿着霍夫曼树一步步完成的，因此这种softmax取名为"Hierarchical Softmax"。
* 返回损失值
*/
real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}
/*
* 计算输出 softmax
*/
void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  output.mul(*wo_, hidden);
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}
/*
* softmax 
*/
real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]);
}
/*
*  input - hidden 计算
*/
void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r) {
  return l.first > r.first;
}
/*
* 预测模型
*/
void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  assert(k > 0);
  heap.reserve(k + 1);
  computeHidden(input, hidden);
  if (args_->loss == loss_name::hs) {
    dfs(k, 2 * osz_ - 2, 0.0, heap, hidden);
  } else {
    findKBest(k, heap, hidden, output);
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap) {
  predict(input, k, heap, hidden_, output_);
}

void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>>& heap,
                      Vector& hidden, Vector& output) const {
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    if (heap.size() == k && log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}
/*
* 深度优先--树的先序遍历
*/
void Model::dfs(int32_t k, int32_t node, real score,
                std::vector<std::pair<real, int32_t>>& heap,
                Vector& hidden) const {
  if (heap.size() == k && score < heap.front().first) {
    return;
  }
  // 叶子节点
  if (tree[node].left == -1 && tree[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }
  // 计算当前哈夫曼树的 real
  real f = sigmoid(wo_->dotRow(hidden, node - osz_));
  dfs(k, tree[node].left, score + log(1.0 - f), heap, hidden);
  dfs(k, tree[node].right, score + log(f), heap, hidden);
}
/*
* 更新梯度
*/
void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;
  computeHidden(input, hidden_);
  if (args_->loss == loss_name::ns) {
    loss_ += negativeSampling(target, lr);
  } else if (args_->loss == loss_name::hs) {
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
    loss_ += softmax(target, lr);
  }
  nexamples_ += 1;

  if (args_->model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addRow(grad_, *it, 1.0);
  }
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs) {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}
/*
* 构建哈夫曼树
*/
void Model::buildTree(const std::vector<int64_t>& counts) {
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  // 输出根节点到叶子节点的路径
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}
/*
* 损失函数
*/ 
real Model::getLoss() const {
  return loss_ / nexamples_;
}
/*
* 初始化 sigmoid
* 为什么 x 值进行了变化 不明白？
*/
void Model::initSigmoid() {
  t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
  }
}
/*
* 初始化 log
*/
void Model::initLog() {
  t_log = new real[LOG_TABLE_SIZE + 1];
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log[i] = std::log(x);
  }
}
/*
* 获取log（x） ，初始化之后，后面使用直接获取
*/
real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int i = int(x * LOG_TABLE_SIZE);
  return t_log[i];
}
/*
* sigmoid 计算， 为了提高计算效率，过大直接返回1.0 过小直接返回 0.0
* t_sigmoid 在 initSigmoid 中初始化之后，后续使用直接取，更加提高了计算效率
* x 转化为 i，直接取，为什么？ 不是明白 
*/
real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

}
