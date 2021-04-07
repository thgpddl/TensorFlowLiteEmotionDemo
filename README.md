# TensorFlowLiteEmotionDemo
在Android上运行人脸表情识别的tflite模型

首先本项目是分类任务，检测任务等需要理解一下代码，但是也是可以使用的

[项目来自于这里](https://blog.doiduoyi.com/articles/1595399632765.html)
对原项目进行了改动，使得你替换的模型的输入不管是[1 * n * n * 3]还是[1 * n * n * 1]都能够正常运行，不过你还需要进行一下的改动

### 如何替换为自己的模型？

#### 1. 替换模型
模型资源文件位于`app\src\main\assets`,在这里你可以替换为自己的tflite模型和相关标签
> 关于模型转换我使用的Keras的H5转tflite，[详见](https://blog.csdn.net/qq_40243750/article/details/115332640)

#### 2. 修改输入输出节点的名称
在TFLiteClassificationUtil.class的构造函数中，有以下四行代码需要注意：
```java
// conv2d_input是输入节点的名称，需要修改为你自己模型的输入节点名称
imageShape = tflite.getInputTensor(tflite.getInputIndex("conv2d_input")).shape();
DataType imageDataType = tflite.getInputTensor(tflite.getInputIndex("conv2d_input")).dataType();

// Identity是输出节点的名称，需要修改为你自己模型的输出节点名称
int[] probabilityShape = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).shape();
DataType probabilityDataType = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).dataType();
```
> 关于如何得知tflite模型的输入输出节点名称，在[TensorFlow Lite(Keras的.H5模型)](https://blog.csdn.net/qq_40243750/article/details/115332640)的2节有介绍和代码实现

#### 3. 已经完成了，开始Build吧！
