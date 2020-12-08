# 面向鹏程公交车司机异常行为数据集的基于通道判别力感知的通道剪枝算法
Discrimination-aware Channel Pruning(DCP)
# 作者
liujing
# 项目简介
## 1. 功能
基于通道判别力感知，实现高效准确的2D模型通道剪枝。
## 2. 性能
|  模型 | Top-1(%) | Top-5(%) | #FLOPs(M) | #Params(M)|
   | ----- |----- |----- | ----- | ----- |
  | ResNet-50 | 95.28| 99.67 | 1787.75 | 10.93 |
  | ResNet-152 | 95.78 | 99.79 | 4390.87 | 22.32 | 

## 3. 评估指标
Top-1，Top-5，#Params，#FLOPs
## 4. 使用数据集
鹏城公交车司机异常行为数据集
## 5. 下载预训练模型
下载[resnet50](https://doc-0o-3c-docs.googleusercontent.com/docs/securesc/nho5chsrj6aihmse7dnqgtjlkukj7sf3/mvfbglj6os2urg81dcdraqgo96s8opha/1606126950000/07924803841725647048/07924803841725647048/1fQXxG3BxKtx66Cl2LCxsOEau2QxQtvoL?e=download&authuser=0&nonce=ij95fov83thfk&user=07924803841725647048&hash=ui98ij299u72nikfem43esq79u9la99t)和[resnet152](https://doc-04-3c-docs.googleusercontent.com/docs/securesc/nho5chsrj6aihmse7dnqgtjlkukj7sf3/gju544fv49akvuro52bfdmlcoln186rv/1606127025000/07924803841725647048/07924803841725647048/1xOeCE4y3SRCZr_m797Bd-bPoV3oVNuHG?e=download&authuser=0)模型文件并存放到dcp/pretrain_models/路径下

# 运行环境与依赖
代码运行的环境与依赖。如下所示：

|类别|名称|版本|
|-----|-----|-----|
|os|ubuntu|16.04|
||python|2.7|
|深度学习框架|pytorch|0.4.0|
|深度学习框架|tensorflow|1.12.0|
||pyhocon|0.3.57|
||prettytable|2.0.0|

# 输入与输出
代码的输入与输出如下所示：
|名称|说明|
|-----|-----|
|输入|剪枝后模型的保存路径|
|输入|数据集的路径|
|输入|预训练模型参数的路径|
|输入|实验ID|
|输入|剪枝率|
|输出|剪枝并微调后的模型|

# 运行方式
在terminal下运行以下命令。
```Shell
# export PYTHONPATH=/path/to/DCP/:$PYTHONPATH
# in my case
export PYTHONPATH=/userhome/DCP/:$PYTHONPATH
python DiscriminationAwareChannelPruning.py -c config.yaml -n 0
```