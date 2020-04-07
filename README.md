# 环境安装及配置
## 1. python && opencv
refers to :https://blog.csdn.net/zhou4411781/article/details/95346578


# 算法介绍
## 算法组成
火灾检测算法主要包括：（1）firenet网络模型构建（2）利用RGB颜色模型过滤条件以及帧差法提取每帧图像关键区域（3）构建生产者和消费者模式读取和检测（4）根据消费中检测结果实时显示检测结果

![firenet 模型结构](https://github.com/YasinChou/small-fire-detect/blob/master/pics_temp/model_struct.jpg)
                                   
                                              图1-Firente模型结构图
![颜色滤波图](https://github.com/YasinChou/small-fire-detect/blob/master/pics_temp/color_filter.png)
                                      
                                              图2-颜色滤波得到的角点图                             

## 检测流程
                                       

具体为（以在线检测单线程为例）：根据多线程得到的图像；对图片图进行RGB颜色滤波，得到模板后进行膨胀，对模板进行角点查找和过滤，其中GRB颜色过滤条件为：

![RGB颜色模型过滤条件](https://github.com/YasinChou/small-fire-detect/blob/master/pics_temp/formule.png)

再根据帧差法去除静止的误检区块；对每帧的区块送入到firenet检测，并存放所有的检测结果中发生火灾的具体位置，如果有一个区块被检测到火灾则该帧视为发生了火灾；根据上个步骤中存放火灾具体位置，实时显示该帧的火灾状况并画出具体火灾位置。


## 相关参数介绍

（1）多边形拟合精度参数，在对轮廓进行多边形拟合时设置的拟合出的多边形与原角点的最大距离，精度设置的越小，拟合出的多边形定点数会越多，时间耗费也较多，目前根据角点周长（arclen）动态设定。

（2）角点长宽比，在角点检测后，因为存在干扰颜色，例如较长的黄色栏杆等，根据长宽比去除。

（3）角点面积上下限，在角点检测后，为了排除面积过小或者过大的检测区域而设置的上、下限。

（4）角点多边形拟合的顶点数，角点多边形拟合后为了除去比较规则的根据颜色提取的区域而设置条件，我们认为火灾区域的角点拟合出的多边形顶点会相对较多。

（5）帧差法中白色区域占比，在检测的区块中，利用帧差法判断该区块是否为静止，如果白色区域占整个纯白图（尺寸是每个检测的区块尺寸）的10%以上，则认为该区域是在变化。

![相关参数](https://github.com/YasinChou/small-fire-detect/blob/master/pics_temp/parameters.png)

## 检测结果示例

检测到火灾结果示例：

![检测到火灾示意图](https://github.com/YasinChou/small-fire-detect/blob/master/pics_temp/fire.png)

未检测到火灾结果示例：

![未检测到火灾示意图](https://github.com/YasinChou/small-fire-detect/blob/master/pics_temp/no_fire.png)

# 运行
 （1）对某个录制视频（161-fire_cut.mp4）进行火灾检测
 
     python fire_detection_color_filter_frame_diff_video.py
 
 （2）对某一摄像头进行在线火灾检测，在代码中可以指定某个摄像头的id（当前代码中是163号摄像头）

     python fire_detection_color_filter_frame_diff_online.py
