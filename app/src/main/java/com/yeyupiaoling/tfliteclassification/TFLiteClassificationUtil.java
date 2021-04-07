package com.yeyupiaoling.tfliteclassification;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import java.io.File;
import java.io.FileInputStream;

public class TFLiteClassificationUtil {
    private static final String TAG = TFLiteClassificationUtil.class.getName();
    private Interpreter tflite;
    private TensorImage inputImageBuffer;
    private final TensorBuffer outputProbabilityBuffer;
    private static final int NUM_THREADS = 4;
    private static final float[] IMAGE_MEAN = new float[]{128.0f, 128.0f, 128.0f};
    private static final float[] IMAGE_STD = new float[]{128.0f, 128.0f, 128.0f};
    private final ImageProcessor imageProcessor;
    int[] imageShape;   //用以储存model的input_shape    [1,n,n,channel]


    /**
     * @param modelPath model path
     */
    public TFLiteClassificationUtil(String modelPath) throws Exception {

        File file = new File(modelPath);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }

        try {
            Interpreter.Options options = new Interpreter.Options();
            // 使用多线程预测
            options.setNumThreads(NUM_THREADS);
            // 使用Android自带的API或者GPU加速
            NnApiDelegate delegate = new NnApiDelegate();
//            GpuDelegate delegate = new GpuDelegate();
            options.addDelegate(delegate);
            tflite = new Interpreter(file, options);
            // 获取输入，shape为{1, height, width, 3}
            imageShape = tflite.getInputTensor(tflite.getInputIndex("conv2d_input")).shape();
            DataType imageDataType = tflite.getInputTensor(tflite.getInputIndex("conv2d_input")).dataType();
            inputImageBuffer = new TensorImage(imageDataType);      //照输入的类型要求，创建TensorImage对象
            // 获取输入，shape为{1, NUM_CLASSES}
            int[] probabilityShape = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).shape();
            DataType probabilityDataType = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).dataType();
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

            // 添加图像预处理方式
            //about input/output more info: https://www.tensorflow.org/lite/inference_with_metadata/lite_support
            imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(imageShape[1], imageShape[2], ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                    .build();
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("load model fail!");
        }
    }

    // 重载方法，根据图片路径转Bitmap预测
    public float[]  predictImage(String image_path) throws Exception {

        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);  //根据图片url创建文件输入流对象
        Bitmap bitmap = BitmapFactory.decodeStream(fis);   //用于从指定输入流中解析、创建 Bitmap 对象
        float[] result = predictImage(bitmap);
        if (bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return result;
    }

    // 重载方法，直接使用Bitmap预测
    public float[] predictImage(Bitmap bitmap) throws Exception {

        return predict(bitmap);
    }


    private ByteBuffer preProcess(Bitmap bitmap) {

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // 设置想要的大小（也即是model的input shape）
        int newWidth = imageShape[1];
        int newHeight = imageShape[2];

        bitmap=Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, false);



        //buffer初始化    //定义一个Buffer,可以直接加载run()
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 1 * newWidth * newHeight * 1);
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();

        // The bitmap shape should be 28 x 28
        int[] pixels=new int[newWidth * newHeight];

        bitmap.getPixels(pixels, 0, newWidth, 0, 0, newWidth, newHeight);


        for(int i =0;i<newWidth * newHeight;i++){
            int pixel = pixels[i];
            //inputBuffer.putFloat((0xff - channel).toFloat());

            float avg = (((pixel >> 16) & 0xFF) * 38 + ((pixel >> 8) & 0xFF) * 75 + (pixel & 0xFF) * 15) >> 7;  //pixels是多字节直接换算后的整数，所以还是用位运算更直接简便

            inputBuffer.putFloat(avg);
            }
        return inputBuffer;


    }

    // 数据预处理
    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);     //TensorImage对象加载bitmap图
        return imageProcessor.process(inputImageBuffer);
    }


    // 执行预测
    private float[] predict(Bitmap bmp) throws Exception {   //传入一张bitmap图

        ByteBuffer inputImageBuffer=null;

        if(imageShape[3]==1){              //如果模型的输入是1channel即灰度图图，则使用preProcess函数处理
            inputImageBuffer= preProcess(bmp);
                            }
        else{
            if(imageShape[3]==3){          //如果模型的输入是3channel即RGB图，则使用loadImage函数处理，并得到buffer
                inputImageBuffer = loadImage(bmp).getBuffer();
                                }
            }




        try {
            tflite.run(inputImageBuffer, outputProbabilityBuffer.getBuffer().rewind());   //开始推理
        } catch (Exception e) {
            throw new Exception("predict image fail! log:" + e);
        }

        float[] results = outputProbabilityBuffer.getFloatArray();
        int l = getMaxResult(results);
        return new float[]{l, results[l]};
    }

    // 获取概率最大的标签
    public static int getMaxResult(float[] result) {

        float probability = 0;
        int r = 0;
        for (int i = 0; i < result.length; i++) {
            if (probability < result[i]) {
                probability = result[i];
                r = i;
            }
        }
        return r;
    }

}
