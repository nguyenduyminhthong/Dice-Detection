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
            DICE_AREA = 400, DOT_AREA = 120, THRESH = 158, MAX_VALUE = 255;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String... args) {
        Mat raw_img = Imgcodecs.imread("src/main/resources/images/Dice.jpg");
        Mat gray_img = raw_img.clone(), blurred_img = raw_img.clone(), binary_img = raw_img.clone(),
                hierarchy = new Mat();

        Imgproc.cvtColor(raw_img, gray_img, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray_img, blurred_img, new Size(5, 5), 0);
        Imgproc.threshold(blurred_img, binary_img, THRESH, MAX_VALUE, Imgproc.THRESH_BINARY);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(binary_img, contours, hierarchy, Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); ++i) {
            if (hierarchy.get(0, i)[2] != -1) {
                RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(contours.get(i).toArray()));
                double area = rect.boundingRect().area();
                if (area > DICE_AREA) {
                    int number_of_dots = 0, child_level = (int) hierarchy.get(0, i)[2];
                    while (child_level != -1) {
                        area = Imgproc
                                .minAreaRect(new MatOfPoint2f(contours.get(child_level).toArray()))
                                .boundingRect().area();
                        if (area > DOT_AREA)
                            ++number_of_dots;
                        child_level = (int) hierarchy.get(0, child_level)[0];
                        Point points[] = new Point[4];
                        rect.points(points);
                        for (int j = 0; j < 4;)
                            Imgproc.line(raw_img, points[j], points[++j % 4], COLOR, THICKNESS,
                                    Imgproc.LINE_AA);
                    }
                    Imgproc.putText(raw_img, number_of_dots + "", rect.center, FONT, SCALE, COLOR,
                            THICKNESS);
                }
            }
        }

        HighGui.imshow("Output", raw_img);
        HighGui.waitKey();
        System.exit(0);
    }
}
