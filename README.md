# 	numpy-realizes-CNN (LeNet5)

​	这是本人的第一个github程序。适用于想要了解CNN基本组成模块的原理及实现。无论您是初学者还是已经了解BP算法，只要您没有自己动手实现过，
那么就推荐您参考、查看。代码有详细的注释+原理链接

## 1、安装：
   * anaconda+pycharm（注python3.0以上）
   * 具体安装网上教程一抓一大把
   
   
## 2、实验结果

   ![实验结果](./pic/Figure_1.png)
   ![实验结果](./pic/Figure_2.png)
   
   * 实验中共迭代20个epoch，在测试集上准确率达到93.52%，当然从曲线可以看出网络并没有完全收敛
   * 接下来的工作，加速网络收敛的同时，进一步提高准确率
   
   
## 3、程序说明（代码中有详细注释）

###  3.1 卷积层
<center>

| con.py           |  使用了for循环，速度低下       |
| :--------------- | :----------------------:   | 
| **con_fast.py**  |     **利用mag2col函数，实现了卷积的快速运算**    |
</center>
 
   * 参考资料：  
   
      
      前向+反向传播过程推导：
      https://blog.csdn.net/qq_16137569/article/details/81449209
      https://blog.csdn.net/qq_16137569/article/details/81477906
      但是这个链接会给出W,b更新公式是错的。
      
   * 说明：为了简化复杂度，默认卷积核均为正方形；行/列方向的步长一样；关于img2col的实现感觉网上讲的都不够直观，可以看“鱼书”中的插图
    （非常直观）


### 3.2 池化层
<center>

|     pooling.py    |  **实现池化层**  |
| :---------------: | :------------: | 
|    *Maxpooling*   |    *最大值池化*  |
| *Averagepooling*  |    *平均池化*   |

</center>

   * 推导过程：https://blog.csdn.net/qq_16137569/article/details/81477906
    实现参考：https://zhuanlan.zhihu.com/p/70713747
   * 说明：为了简化问题，默认池化层的核均为正方形；核的尺寸与步长一样


### 3.3 全连接层：fc.py
   * 全连接层的实现比较简单，（NG的视频上已经非常清楚）。有两种略有不同的实现方式但是原理上是一样的。我会写个知乎解释：
   * 知乎链接----------------（敬请期待）

### 3.4 激活层
<center>

| activate.py    |     实现各种激活函数    |
| :------------- | :-----------------:  | 
|     *ReLu*     |     *y=max(0, x)*    |
|    *sigmoid*   |     *y=1/(1+e^(-x))*    |
|     *tanh*     |     *y=(e^x-e^(-x)) / (e^x+e^(-x))*    |
</center>

   * 很简单，是所有模块中实现最简单的一个


### 3.5 损失函数: loss.py
<center>

|     loss.py    |     实现各种损失函数    |
| :------------- | :-----------------:  | 
|    *softmax*   |     *y=-ylna*    |
|      *ESM*     |     *敬请期待*    |
| *Cross-entropy*|     *敬请期待*    |
</center>

   * 推导过程：https://zhuanlan.zhihu.com/p/67759205
   * 实现参考：https://blog.csdn.net/QLBFA/article/details/107536486
   * 说明：目前只实现了一个softmax，后续会加入：均方差损失（ESM）、交叉熵损失（Cross-entropy）

## 4、实现LeNet5

   ![Alt text-w518-h158](./pic/LeNet5.png)
   * 注：我的网络实现时，输入的图像尺寸为（1x28x28），所以第一个卷积采用了SAME填充方式，其他结构均与原论文一致。
<center>

|     net/LeNet5.py    |     实现LeNet网络结构+forward/backward    |
| :------------------: | :------------------------------------:  | 
|      **train.py**    |              **完成训练+测试**                |
</center>

  

## 附录：  
https://github.com/leeroee/NN-by-Numpy  （函数接口参考）  
https://github.com/wuziheng/CNN-Numpy  （实现参考）

>>> 地球人都爱猫，这是大自然定的---------沃兹基硕德
   ![cat](./pic/cat.jpg)


