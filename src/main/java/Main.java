package main.java;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {
    private static final Scalar COLOR = new Scalar(255, 255, 0, 0);
    private static final int FONT = Imgproc.FONT_HERSHEY_SIMPLEX, SCALE = 1, THICKNESS = 3,
            DICE_AREA = 400, DOT_AREA = 120, THRESH = 158, MAX_VALUE = 255, SIGMA = 0, STATUS = 0;
    private static final Size KERNEL_SIZE = new Size(5, 5);

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String... args) {
        Mat rawImg = Imgcodecs.imread("src/main/resources/images/Dice.jpg");
        Mat grayImg = rawImg.clone(), blurredImg = rawImg.clone(), binaryImg = rawImg.clone();

        Imgproc.cvtColor(rawImg, grayImg, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(grayImg, blurredImg, KERNEL_SIZE, SIGMA);
        Imgproc.threshold(blurredImg, binaryImg, THRESH, MAX_VALUE, Imgproc.THRESH_BINARY);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binaryImg, contours, hierarchy, Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_SIMPLE);

        for (int row = 0, column = 0; column < contours.size(); ++column) {
            if (hierarchy.get(row, column)[2] != -1) {
                RotatedRect rect =
                        Imgproc.minAreaRect(new MatOfPoint2f(contours.get(column).toArray()));
                double area = rect.boundingRect().area();
                if (area > DICE_AREA) {
                    int numberOfDots = 0, childLevel = (int) hierarchy.get(row, column)[2];
                    while (childLevel != -1) {
                        area = Imgproc
                                .minAreaRect(new MatOfPoint2f(contours.get(childLevel).toArray()))
                                .boundingRect().area();
                        if (area > DOT_AREA)
                            ++numberOfDots;
                        childLevel = (int) hierarchy.get(row, childLevel)[0];
                        Point points[] = new Point[4];
                        rect.points(points);
                        for (int i = 0; i < 4;)
                            Imgproc.line(rawImg, points[i], points[++i % 4], COLOR, THICKNESS,
                                    Imgproc.LINE_AA);
                    }
                    Imgproc.putText(rawImg, numberOfDots + "", rect.center, FONT, SCALE, COLOR,
                            THICKNESS);
                }
            }
        }

        HighGui.imshow("Output", rawImg);
        HighGui.waitKey();
        System.exit(STATUS);
    }
}
