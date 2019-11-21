import java.io.*;


import java.awt.*;
import javax.swing.*;

public class MNISTReader extends JFrame {

    private static int m_z = 12345, m_w = 45678;

    private static final double LEARNING_RATE = 0.1;
    private static final int NEURONS = 28 * 28 + 10; // pixel_length * pixel_width + special for labels
    private static final int MAX_PATTERNS = 1000;
    private int numLabels;
    private int numImages;
    private int numRows;
    private int numCols;

    private double trainLabels[] = new double[MAX_PATTERNS];
    private double trainImages[][] = new double[MAX_PATTERNS][28 * 28];
    private double weights[][] = new double[NEURONS][NEURONS]; // all neurons are connected with each other
    private double output[] = new double[NEURONS];
    private double input[] = new double[NEURONS];
    private double reconstructed_input[] = new double[NEURONS];

    /**
     * @param args args[0]: label file; args[1]: data file.
     * @throws IOException
     * @throws InterruptedException
     */
    public static void main(String[] args) throws IOException, InterruptedException {

        MNISTReader frame = new MNISTReader();
        frame.readMnistDatabase();
        frame.setSize(900, 350);

        System.out.println("Learning step:");
        frame.trainOrTestNet(true, 10000, frame);

        System.out.println("Teststep:");
        frame.trainOrTestNet(false, 1000, frame);
    }

    /**
     * perform bolzmann steps on the network as training or test step
     *
     * @param train    flag
     * @param maxCount maximum of tests
     * @param frame    frame
     */
    private void trainOrTestNet(boolean train, int maxCount, MNISTReader frame) {

        int correct = 0;

        if (train) {
            init(weights);
        }
        int pattern = 0;

        for (int count = 1; count < maxCount; count++) {
            // --- training phase

            for (int t = 0; t < NEURONS - 10; t++) {
                input[t] = trainImages[pattern % 100][t]; // initialize original pattern
            }
            for (int t = NEURONS - 10; t < NEURONS; t++) {
                input[t] = 0;
            }
            if (train) {
                // --- use the label also as input!
                if (trainLabels[pattern % 100] >= 0 && trainLabels[pattern % 100] < 10) {
                    input[NEURONS - 10 + (int) trainLabels[pattern % 100]] = 1.0;
                }
            }

//          drawActivity(0, 0, input, red, green, blue);

            // --- Contrastive divergence
            // Activation
            input[0] = 1;                    // bias neuron!
            activateForward(input, weights, output); // positive Phase
            output[0] = 1;                    // bias neuron!

//			drawActivity(300,0,output,red,green,blue);

            activateReconstruction(reconstructed_input, weights, output); // negative phase/ reconstruction

//			drawActivity(600,0,reconstructed_input,red,green,blue);
            if (train) {
                contrastiveDivergence(input, output, reconstructed_input, weights);
            }

            if (count % 111 == 0) {
                System.out.println("Zahl:" + trainLabels[pattern % 100]);
                System.out.println("Trainingsmuster:" + count + "                 Erkennungsrate:" + (float) (correct) / (float) (count) + " %");
                frame.validate();
                frame.setVisible(true);
                frame.repaint();
                try {
                    Thread.sleep(20); //20 milliseconds is one second.
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                }
            }

            if (!train) {
                int number = 0;
                for (int t = NEURONS - 10; t < NEURONS; t++) {
                    if (reconstructed_input[t] > reconstructed_input[NEURONS - 10 + number]) {
                        number = t - (NEURONS - 10);
                    }
                }

                if (frame.trainLabels[pattern % 100] == number) {
                    System.out.println("Muster: " + frame.trainLabels[pattern % 100] + ", Erkannt: " + number + " KORREKT!!!\n");
                    correct++;
                } else {
                    System.out.println("Muster: " + frame.trainLabels[pattern % 100] + ", Erkannt: " + number);
                }

            }

            pattern++;
        }

    }

    /**
     * init the weights randomly
     *
     * @param weights weights
     */
    private void init(double weights[][]) {
        for (int t = 0; t < NEURONS; t++) {
            for (int neuron = 0; neuron < NEURONS; neuron++) {
                weights[neuron][t] = randomGen() % 2000 / 1000.0 - 1.0;
                //	System.out.println("weight["+neuron+"]["+t+"]="+weights[neuron][t]);
            }
        }
    }

    private double sigmoid(double x) {
        return  1 / ( 1 + Math.pow(Math.E, (-1 * x)));
    }

    private double relu(double x) {
        return x > 0 ? x : 0;
    }

    /**
     * activates the output layer using weights and the inputs
     * writes into out
     *
     * @param in  input layer
     * @param w   weights
     * @param out output layer
     */
    private void activateForward(double in[], double w[][], double out[]) {

        for (int j = 0; j < out.length; j++) {

            double activity = 0;

            for (int i = 0; i < in.length; i++) {
                activity += (w[j][i] * in[i]);
            }

            out[j] = sigmoid(activity);

        }


    }

    /**
     * reconstructs the input layer using the weights and outputs
     * writes into rec
     *
     * @param rec reconstruction layer
     * @param w   weights
     * @param out output layer of forwarding before ( -> hidden layer)
     */
    private void activateReconstruction(double rec[], double w[][], double out[]) {

        for (int i_rec = 0; i_rec < rec.length; i_rec++) {

            double rec_activity = 0;

            for (int j = 0; j < out.length; j++) {
                rec_activity += (w[i_rec][j] * out[j]);
            }

            rec[i_rec] = sigmoid(rec_activity);

        }

    }

    /**
     * applies the restricted bolzmann machine learning rule
     * writes into w
     *
     * @param inp input layer v0
     * @param out hidden layer h
     * @param rec generated input layer v1
     * @param w   weights
     */
    private void contrastiveDivergence(double inp[], double out[], double rec[], double w[][]) {

        // < rec.length also possible
        for (int i = 0; i < inp.length; i++) {
            for (int j = 0; j < out.length; j++) {
                w[j][i] += (LEARNING_RATE * (out[j] * inp[i] - out[j] * rec[i]));
            }
        }

    }

    /**
     * reads mnist from files
     *
     * @throws IOException
     */
    private void readMnistDatabase() throws IOException {

        DataInputStream labels = new DataInputStream(new FileInputStream(
                "train-labels-idx1-ubyte"));
        DataInputStream images = new DataInputStream(new FileInputStream(
                "train-images-idx3-ubyte"));
        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.println("Label file has wrong magic number: "
                    + magicNumber + " (should be 2049)");
            System.exit(0);
        }
        magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.println("Image file has wrong magic number: "
                    + magicNumber + " (should be 2051)");
            System.exit(0);
        }
        numLabels = labels.readInt();
        numImages = images.readInt();
        numRows = images.readInt();
        numCols = images.readInt();

        long start = System.currentTimeMillis();
        int numLabelsRead = 0;
        int numImagesRead = 0;

        while (labels.available() > 0 && numLabelsRead < MAX_PATTERNS) {// numLabels

            byte label = labels.readByte();
            numLabelsRead++;
            trainLabels[numImagesRead] = label;
            double pos = 0, neg = 0;
            int i = 0;
            for (int colIdx = 0; colIdx < numCols; colIdx++) {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    if (images.readUnsignedByte() > 0) {
                        trainImages[numImagesRead][i++] = 1.0;
                    } else {
                        trainImages[numImagesRead][i++] = 0;
                    }

                }
            }

            numImagesRead++;

            // At this point, 'label' and 'image' agree and you can do
            // whatever you like with them.

            if (numLabelsRead % 10 == 0) {
                System.out.print(".");
            }
            if ((numLabelsRead % 800) == 0) {
                System.out.print(" " + numLabelsRead + " / " + numLabels);
                long end = System.currentTimeMillis();
                long elapsed = end - start;
                long minutes = elapsed / (1000 * 60);
                long seconds = (elapsed / 1000) - (minutes * 60);
                System.out
                        .println("  " + minutes + " m " + seconds + " s ");

            }

        }

        System.out.println();
        long end = System.currentTimeMillis();
        long elapsed = end - start;
        long minutes = elapsed / (1000 * 60);
        long seconds = (elapsed / 1000) - (minutes * 60);
        System.out.println("Read " + numLabelsRead + " samples in "
                + minutes + " m " + seconds + " s ");

        labels.close();
        images.close();

    }

    /**
     * returns a random
     *
     * @return random number
     */
    private int randomGen() {
        m_z = Math.abs(36969 * (m_z & 65535) + (m_z >> 16));
        m_w = Math.abs(18000 * (m_w & 65535) + (m_w >> 16));
        return Math.abs((m_z << 16) + m_w);
    }

    public void paint(Graphics g) {

        int i = 0;
        for (int colIdx = 0; colIdx < 28; colIdx++) {
            for (int rowIdx = 0; rowIdx < 28; rowIdx++) {
                int c = (int) (input[i++]);
                if (c > 0.0) {
                    g.setColor(Color.blue);
                } else {
                    g.setColor(Color.black);
                }

                g.fillRect(10 + rowIdx * 10, 10 + colIdx * 10, 8, 8);
            }
        }

        for (int t = 0; t < 10; t++) {
            int c = (int) (input[i++]);
            if (c > 0.0) {
                g.setColor(Color.blue);
            } else {
                g.setColor(Color.black);
            }
            g.fillRect(10 + t * 10, 10 + 28 * 10, 8, 8);
        }

        i = 0;
        for (int colIdx = 0; colIdx < 28; colIdx++) {
            for (int rowIdx = 0; rowIdx < 28; rowIdx++) {
                int c = (int) (output[i++] + 0.5);
                if (c > 0.0) {
                    g.setColor(Color.blue);
                } else {
                    g.setColor(Color.black);
                }

                g.fillRect(300 + 10 + rowIdx * 10, 10 + colIdx * 10, 8, 8);
            }
        }

        for (int t = 0; t < 10; t++) {
            int c = (int) (output[i++] + 0.5);
            if (c > 0.0) {
                g.setColor(Color.blue);
            } else {
                g.setColor(Color.black);
            }
            g.fillRect(300 + 10 + t * 10, 10 + 28 * 10, 8, 8);
        }

        i = 0;
        for (int colIdx = 0; colIdx < 28; colIdx++) {
            for (int rowIdx = 0; rowIdx < 28; rowIdx++) {
                int c = (int) (reconstructed_input[i++] + 0.5);
                if (c > 0.0) {
                    g.setColor(Color.blue);
                } else {
                    g.setColor(Color.black);
                }

                g.fillRect(600 + 10 + rowIdx * 10, 10 + colIdx * 10, 8, 8);
            }
        }

        for (int t = 0; t < 10; t++) {
            int c = (int) (reconstructed_input[i++] + 0.5);
            if (c > 0.0) {
                g.setColor(Color.blue);
            } else {
                g.setColor(Color.black);
            }
            g.fillRect(600 + 10 + t * 10, 10 + 28 * 10, 8, 8);
        }

    }

	/*
	  public static void writeFile(String fileName, byte[] buf)
	    {
			
			FileOutputStream fos = null;
			
			try
			{
			   fos = new FileOutputStream(fileName);
			   fos.write(buf);
			}
			catch(IOException ex)
			{
			   System.out.println(ex);
			}
			finally
			{
			   if(fos!=null)
			      try
			      {
			         fos.close();
			      }
			      catch(Exception ex)
			      {
			      }
			}
	    }
*/
}
