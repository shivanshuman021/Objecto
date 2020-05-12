package com.mishra.objecto;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.database.Cursor;
import android.Manifest;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE =1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonLoadImage =  findViewById(R.id.button);
        Button detectButton = findViewById(R.id.detect);

        if (Build.VERSION.SDK_INT >=Build.VERSION_CODES.M){
            requestPermissions(new String[]{
                    Manifest.permission.READ_EXTERNAL_STORAGE},1);

        }

        buttonLoadImage.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View arg0){
                TextView textView = findViewById(R.id.result_text);
                textView.setText("");
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(i,RESULT_LOAD_IMAGE);

            }
        });


        detectButton.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View arg0){
                Bitmap bitmap = null;
                Module module = null;

                //Getting the image from image view
                ImageView imageview =  findViewById(R.id.image);

                try{
                    //Read image as bitmap
                    bitmap = ((BitmapDrawable) imageview.getDrawable()).getBitmap();


                    bitmap = Bitmap.createScaledBitmap(bitmap,224,224,true);

                    //Loading model file
                    module = Module.load(fetchModelFile(MainActivity.this,"resnet18_traced.pt"));
                }

                catch (IOException e){

                    Log.e("PytorchHelloWorld", "Error reading assets", e);
                    finish();
                }

                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                //Calling the forward of the model to run our input

                final Tensor output = module.forward(IValue.from(input)).toTensor();

                final float[] score_arr = output.getDataAsFloatArray();

                //fetch index of the value with max score
                float max_score = -Float.MAX_VALUE;
                int ms_ix = -1;
                for (int i=0; i<score_arr.length ;i++){
                    if (score_arr[i]>max_score){
                        max_score = score_arr[i];
                        ms_ix = i;
                    }
                }

                //fetching the name from the list based on the index
                String detected_class = ModelClasses.MODEL_CLASSES[ms_ix];

                //writing the detected class in to the text view of the layout
                TextView textView = findViewById(R.id.result_text);
                textView.setText(detected_class);

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode,int resultCode,Intent data){
        //this function returns selected image from gallery
        super.onActivityResult(requestCode,resultCode,data);

        if (requestCode==RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data){
            Uri selectedImage =data.getData();
            String[] filePathColumn = {MediaStore.Images.Media.DATA};


            Cursor cursor = getContentResolver().query(selectedImage,filePathColumn,null,null,null);

            cursor.moveToFirst();

            int coloumnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(coloumnIndex);
            cursor.close();

            ImageView imageView = findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //setting URI so that we can read Bitmap from Image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);

        }
    }


    public static String fetchModelFile(Context context,String modelName) throws IOException{
        File file = new File(context.getFilesDir(),modelName);
        if (file.exists() && file.length()>0){
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)){
            try (OutputStream os = new FileOutputStream(file)){
                byte[] buffer = new byte[4*1024];
                int read;
                while ((read = is.read(buffer)) != -1){
                    os.write(buffer,0,read);
                }
                os.flush();
            }

            return file.getAbsolutePath();
        }
    }
}
