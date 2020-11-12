# 	numpy-realizes-CNN

​	这是本人的第一个github程序。适用于想要了解CNN基本组成模块的原理及实现。无论您是初学者还是已经了解BP算法，只要您没有自己动手实现过，
那么就推荐您参考、查看。代码有详细的注释+原理链接

## 1、安装：
   * anaconda+pycharm（注python3.0以上）
   * 具体安装网上教程一抓一大把
   
   
## 2、实验结果
    
    
## 3、程序说明（代码中有详细注释）

###  3.1 卷积
<center>
<img src="fig/iteration.jpg" style="zoom:60%"/>

| con.py           |  使用了for循环，速度低下    |
| :--------------- | :----------------------: | 
| **con_fast.py**  |     利用mag2col函数，实现了卷积的快速运算    |
</center>
 
   * 参考资料：  
   
      
      前向+反向传播过程推导：
      https://blog.csdn.net/qq_16137569/article/details/81449209
      https://blog.csdn.net/qq_16137569/article/details/81477906
      但是这个链接会给出W,b更新公式是错的。
      
   * 说明：为了简化复杂度，默认卷积核均为正方形；行/列方向的步长一样；关于img2col的实现感觉网上讲的都不够直观，可以看“鱼书”中的插图
    （非常直观）


### 3.2 池化：pooling.py
<center>
<img src="fig/iteration.jpg" style="zoom:60%"/>

|     pooling.py    |  **实现池化层**  |
| :---------------: | :------------: | 
|    *Maxpooling*   |    *最大值池化*  |
| *Averagepooling*  |    *平均池化*   |

</center>

   * 推导过程：https://blog.csdn.net/qq_16137569/article/details/81477906
    实现参考：https://zhuanlan.zhihu.com/p/70713747
   * 说明：为了简化问题，默认池化层的核均为正方形；核的尺寸与步长一样


### 3.3 全连接层：fc.py。
   * 全连接层的实现比较简单，（NG的视频上已经非常清楚）。有两种略有不同的实现方式但是原理上是一样的。我会写个博客解释：
   * 链接-------

### 3.4 激活函数：activate.py。
   * 实现的激活函数包括：Relu, sigmoid, tanh
   * 很简单，是所有模块中实现最简单的一个

### 3.5 损失函数: loss.py
   * 实现的损失函数包括：softmax
   * 推导过程：https://zhuanlan.zhihu.com/p/67759205
   * 实现参考：https://blog.csdn.net/QLBFA/article/details/107536486
