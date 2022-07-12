# C1 绪论

## 1.2 基本术语

**数据集Dataset**  
= {x_1,x_2,...,x_m}  
= {(样本sample/示例instance),...} 每个D有m个样本   
= {(属性attribute/特征feature = 属性值value)，...} 每个样本有d个属性

* 例: 瓜田 = {瓜1,瓜2} = {(色=白;根=硬;声=浊),(色=黑;根=卷;声=脆)}


**样本空间X**  
假设样本符合独立同分布independent & identically distributed(i.i.d)
单个向量x_i = (x_i1,x_i2,...x_id) ∈ X (d-dimentionality维数/属性总数)
单个属性值x_ij ∈ X (i-样本序数,j-属性序数)

* 例: x_1 = 瓜1, x_23 = 瓜2属性3 = 脆 


**建立模型/学习器learner:**  
1. 训练training
    - 对象: training set =  2/3 sample
    - 目标: 得到假设hypothesis(=真实?)
    - 过程: input空间X, output空间Y的映射f(X→Y)
    - 方法:
        - 监督学习supervised learning +lable
            - 分类classification: 预测离散值
                - 二分类binary: 正P/负N,-1/+1,0/1
                - 多分类multi-class: |Y| > 2
            - 回归regression: 预测连续值 Y = R实数集
        - 无监督学习unsupervised learning NO_lable
            - 聚类clustering
    - 结果: 得到假设
2. 测试testing
    - 对象: testing set = 1/3 sample
    - 目标: 强泛化能力generalization


## 1.3 假设空间

**演绎deduction**: 一般→特殊, 上至下top-down, 特化specialization  
**归纳induction**: 特殊→一般, 下至上bottom-up, 泛化generalization  
   - 概念学习concept: Boolean 0/1
   - 黑箱模型


## 1.4 归纳偏好 inductive bias

**一般原则**  
奥卡姆剃刀Occam's razor: 选最简单的, **难点**是定义"简单" by how?

**No free lunch**  
    - 前提: 所有问题出现的机会=, 所有问题的重要性=  
    - 原理: 对部分问题,算法a优于算法b;必然存在另一部分问题,在那里算法b优于算法a.对f均匀分布,则有一半f对x的预测与h(x)不一致,即对两种算法的期望相同.  
    - 结论：关键让自身归纳偏好fit问题
    
$$
\underset{f}{\Sigma} E_{ote}(算法a|X,f) = \underset{f}{\Sigma} E_{ote}(算法b|X,f)
$$


# C2 模型评估与选择  

## 2.1 经验误差与过度拟合

**错误率 error rate  
精度 accuracy  
实际-样本误差 error  
Train: 训练误差 training error/经验误差empirical error  
Test: 泛化误差 generalizaiton error  
过度拟合 overfitting:** 无法避免,只能缓解  
**欠拟合 underfitting:** (解决)决策树-拓展分支;神经网络-增加训练轮数  

## 2.2 模型评估(选择估计方法)

**目标:** 选test泛化误差最小的  
**前提:** 独立同分布,train/test样本互斥  
**方法:** 原始数据量充足时,优先考虑留出法,交叉验证法

### 2.2.1 留出法 hold-out

Dataset = train + test  
1. 分层采样stratified sampling: 单次结果不够稳定
2. 前后随机: 随机划分,重复进行,取平均值作为结果

### 2.2.2 交叉验证法 k-fold cross validation

Dataset = k个子集,每轮有k-1个train和1个test(抽取test不重复),重复k次,结果取均值  
k = 5,**10**,20 常用10

+图  

**特例:** 留一法LOO(leave-one-out) 

D有m个样本,k = m,有且只有唯一方式划分子集,train有k-1个样本=m-1=D-1  
优: 准确度高  
缺: 数据量大难处理; 结果未必永远准确 ∵No free lunch  

### 2.2.3 自助法 bootstraping

train = D'= 随机抽样D[:] * m次  
test = D\D' = D * 1/e ≈ D * 0.368 包外估计out-of-bag estimate  
优: 适用于数据量小,难以有效划分的情况  
缺: 随机抽样改变原始分布,会导致估计偏差  

### 2.2.4 调参,最终模型

Dataset_P1-1: 训练集training set: 训练模型  
Dataset_P1-2: 验证集 validation set: 选最优模型的参数,考虑参数范围和变化步长,折中调参  
Dataset_P2: test测试集

## 2.3 性能度量(选择估计标准)

**回归任务:** 均值方差 mean squared error  
**分类任务:** 如下

### 2.3.1 错误率, 精度

1 = 错误率error rate + 精度accuracy

### 2.3.2 查准率, 查全率, F1

Sample = TP + FP + TN + FN  

+图  

**查准率precision** = TP / (TP + FP)  
**查全率recall** = TP / (TP + FN)  
查全高,查准低; 查准高,查全低.  

+P-R图  

**综合考虑查准&查全的度量:**
1. **平衡点BEP**(break-even point): 查全 = 查准
 - A包C: A优于C  
 - A交B: 比较曲线下的面积大小by平衡点,A优于B  
2. **F1**  
调和平均: F1 = 2 * P * R / (P + R) = 2 * TP / (样例总数 + TP - TN)  
加权平均: Fβ = (1+β^2) * P * R / (β^2 * P) + R, 其中β>0  
    * β＜1: 查准更重要  
    * β= 1: F1  
    * β＜0: 查全更重要
    
   如果有多个二分类混淆矩阵:
    * macro-F1: 先求查全查准, 再取元素均值  
    * micro-F1: 先取元素均值,再求查全查准  

### 2.3.3 ROC, AUC (略)

后面学了补

### 2.3.4 代价敏感错误率(略)

后面学了补

假设: 非均等代价unequal cost: 不同类型错误造成不同损失  
目标: minimize total cost

## 2.4 比较检验

考虑:
- 测试集性能 不一定= 泛化性能
- 测试集性能 结果不同 (∵测试样例不同)
- 算法本身随机性

### 2.4.1 假设检验

#### 二项检验

#### t-检验(t分布)

### 2.4.2 交叉验证t检验(t分布)(2个算法)

思想: 两个模型性能相同,使用相同train/test set得到的错误率相同  
假设: 测试错误率是泛化错误率的独立采样  
 - 不独立原因: 交叉验证时训练集会有重叠→非独立采样→过高估计假设成立的可能  
 - 不独立解决：5次 * 2折交叉验证,2折避免数据重叠,均值只取第1次的结果差,方差5 * 2 全部取

### 2.4.3 McNemar检验(卡方分布)(2个算法)

假设: 两个模型性能相同,e01 = e10, |e01 - e10| 服从卡方分布

### 2.4.4 Friedman检验, Nemenyi后续检验(n个算法)

1. 原始Friedman检验: 根据性能由好到坏排序,赋予序数值,性能相同序值相同,求每一个序的均值.  
   假设:性能相同,平均序值相同,服从卡方分布  
2. 变量Friedman检验: F分布  
3. Nemenyi后续检验post-hoc test: 拒绝原假设"所有算法性能相同"后,进一步区分各个算法
   计算平均序值差的临界值域, 超出-拒绝原假设-显著不同 
   

## 2.5 偏差, 方差  

**偏差方差分解 bias-variance decomposition**: 解释算法的泛化性能

**泛化误差:**
1. 偏差: 期望-真实的差距, 表示学习算法的能力(拟合度)
2. 方差: 训练集变动导致的变化,表示数据的充分性
3. 噪声: 期望泛化误差的min,表示学习任务本身的难度  

ideal泛化误差: (偏差小,方差小)→泛化误差小  

+图

前期: 偏差主导,通过增加训练提高拟合度,使偏差趋小;  
中期: balance;  
后期: 方差主导,训练数据变动影响大, 如包含了trainingset的特性可能导致过度拟合.  

# Task1 学习心得
1. GitHub+Markdown+LaTeX+图=下次一定CSDN;
2. 绪论介绍逻辑不够清晰，整理笔记需要花时间理思路;
3. 基本术语各种或称,或来或去,虽全但晕;时而模型时而学习器,串来串去,我不喜欢;
4. 需要回忆一下统计学检验的内容.
