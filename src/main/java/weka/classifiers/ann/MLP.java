/*
 *    MLP.java
 *    Copyright (C) 2015 by Andre Susanto, Adhika Sigit, Michael Alexander
 *    
 *    Implementation of MULTI LAYER PERCEPTRON Classifier Algorithm
 */

package weka.classifiers.ann;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Vector;
import java.util.Enumeration;

public class MLP extends Classifier implements OptionHandler, WeightedInstancesHandler {
    ////// DATA YANG BERKAITAN DENGAN MODEL
    private float learningRate = 0.1f;
    private int maxIteration = 30;
    private double working_h[];
    private float weights[][];
    private float weights_out[][];
    private int hiddenPerceptrons = 1;
    private int outPerceptron = 1;
    ///////////////////////////////////////
    private NominalToBinary nominalToBinary = new NominalToBinary();
    private Normalize normalize = new Normalize();
    //////////////////////////////////////


    // untuk serialisasi
    private static final long serialVersionUID = -5990607817048210779L;

    private double[] nominalize(double in){
        double out[] = new double[outPerceptron];

        if (outPerceptron == 1){
            out[0] = in;
        }else{
            for (int i = 0; i < outPerceptron; i++){
                out[i] = 0.01;
            }
            out[(int)in] = 0.99;
        }
        return out;
    }

    private double calculateError(Instances instances) throws Exception{
        double tmp_error = 0;
        int sumInstances = instances.numInstances();

        for (int i = 0; i < sumInstances ; i++){
            tmp_error += Math.pow( instances.instance(i).classValue() - classifyInstance(instances.instance(i)) , 2);
        }
        return tmp_error / 2;
    }

    public void buildClassifier(Instances _instances) throws Exception {
        Instances instances;
        getCapabilities().testWithFail(_instances);

        nominalToBinary.setInputFormat(_instances);
        instances = Filter.useFilter(_instances, nominalToBinary);
        normalize.setInputFormat(instances);
        instances = Filter.useFilter(instances, normalize);

        int i = 0; int it = 0;
        int sumInstances = instances.numInstances();
        int sumAttributes = instances.numAttributes();
        int iterateTo = maxIteration * sumInstances;

        if (sumInstances > 0) {
            weights = new float[hiddenPerceptrons][];
            for (int k = 0 ; k < hiddenPerceptrons; k++)
                weights[k] = new float[sumAttributes];

            if (instances.classAttribute().isNominal())
                outPerceptron = instances.classAttribute().numValues();

            System.out.printf("NUM OUT: %d\n", outPerceptron);

            weights_out = new float[outPerceptron][];
            for (int j = 0; j < outPerceptron; j++)
                weights_out[j] = new float[hiddenPerceptrons];


            double curError = 1;

            while (it < iterateTo && curError > 0) {
                double target[] = nominalize(instances.instance(i).classValue());
                double out = classifyInstance(instances.instance(i));


                //System.out.printf("Iterasi %d: (TARGET: %f, OUT: %f)\n", i, instances.instance(i).classValue(), out);
                //System.out.printf("   NEW WEIGHT: ");

                double out_error[] = new double[outPerceptron];
                double hidden_error[] = new double[hiddenPerceptrons];

                for (int l = 0; l < outPerceptron; l++) {
                    out_error[l] = sigma_out[l] * (1 - sigma_out[l]) * (target[l] - sigma_out[l]);
                }

                for (int k = 0; k < hiddenPerceptrons; k++) {
                    double sig_error = 0;
                    for (int l = 0; l < outPerceptron; l++) {
                        sig_error += weights_out[l][k] * out_error[l];
                    }
                    hidden_error[k] = working_h[k] * (1 - working_h[k]) * sig_error;
                }


                for (int l = 0; l < outPerceptron; l++) {
                    for(int j = 0; j < hiddenPerceptrons; j++){
                        //System.out.printf("  --> O %d,%d  DELTA = %f \n", l, j, (learningRate * out_error[l] * working_h[j]));
                        weights_out[l][j] += learningRate * out_error[l] * working_h[j];
                    }
                }

                for(int k = 0; k < hiddenPerceptrons; k++){
                    for (int j = 0; j < sumAttributes; j++) {
                        //System.out.printf("  --> H %d,%d  DELTA = %f \n", k, j, (learningRate * hidden_error[k] * instances.instance(i).value(j)));
                        weights[k][j] += learningRate * hidden_error[k] * instances.instance(i).value(j);
                    }
                }

                /*
                for (int k = 0; k < hiddenPerceptrons; k++) {
                    for (int j = 0; j < sumAttributes; j++) {
                        if (j != instances.classIndex()) {
                            double delta_error = 0;
                            for (int l = 0; l < outPerceptron; l++) {
                                delta_error += (sigma_out[l] - target[l]) * sigma_out[l] * (1.0f - sigma_out[l]) * weights_out[l][k];
                            }
                            double delta_in = delta_error * working_h[k] * (1 - working_h[k]) * instances.instance(i).value(j);
                            weights[k][j] -= learningRate * delta_in;

                            //System.out.printf("\n    W(%d,%d) = %f, DELTA = %f \n", k, j, weights[k][j], delta_in);
                        }
                    }

                    for (int l = 0; l < outPerceptron; l++) {
                        double delta = (sigma_out[l] - target[l]) * sigma_out[l] * (1.0f - sigma_out[l]) * working_h[k];
                        weights_out[l][k] -= learningRate * delta;
                        //System.out.printf("\n    W(%d,OUT %d) = " + weights_out[l][k] + " (%f - %f) DELTA = %f \n", k, l, target[l], sigma_out[l], delta);
                    }


                }*/

                curError = calculateError(instances);
                //System.out.printf("\n   Error: %f\n\n", curError);

                i = (++i) % sumInstances;
                it++;
            }
        }
    }

    private double sigmoid(double in){
        return (1/( 1 + Math.pow(Math.E,(-1* in))));
    }

    private double sigma_out[];

    public double classifyInstance(Instance _instance) throws Exception{
        Instance instance;

        nominalToBinary.input(_instance);
        instance = nominalToBinary.output();
        normalize.input(instance);
        instance = normalize.output();

        int numAttr = instance.numAttributes();
        sigma_out = new double[outPerceptron];
        working_h = new double[hiddenPerceptrons];

        for ( int i = 0 ; i < hiddenPerceptrons ; i++){
            double sigma = 0;
            for ( int j = 0 ; j < numAttr ; j++){
                if (j != instance.classIndex())
                    sigma += weights[i][j] * instance.value(j);
            }

            working_h[i] = sigmoid(sigma);

            for (int l = 0; l < outPerceptron ; l++){
                sigma_out[l] += weights_out[l][i] * working_h[i];
            }
        }

        if (outPerceptron > 1){
            int maxIndex = 0; double maxVal = 0;
            for (int l = 0; l < outPerceptron ; l++){
                sigma_out[l] = sigmoid(sigma_out[l]);
                //System.out.printf("    --> OUT %d = %f \n", l, sigma_out[l]);

                if (l == 0 || maxVal < sigma_out[l]){
                    maxIndex = l;
                    maxVal = sigma_out[l];
                }
            }

            return  maxIndex;
        }else {
            return sigmoid(sigma_out[0]);
        }
    }


    public Enumeration listOptions() {

        Vector newVector = new Vector(14);

        newVector.addElement(new Option(
                "\tNumber of Hidden Perceptrons in Hidden Layer.\n"
                        +"\t(Value should be >= 1.",
                "H", 1, "-H <activation function>"));
        newVector.addElement(new Option(
                "\tMaximum Iteration that is used by PTR.\n"
                        +"\t(Default 30).",
                "M", 1, "-M <max iteration>"));
        newVector.addElement(new Option(
                "\tLearning Rate.\n"
                        +"\t(Default = 0.1).",
                "L", 1,"-L <learning rate>"));

        return newVector.elements();
    }

    public String maxIterationTipText() {
        return "Maximum Iteration that is used by PTR. Default = 30";
    }

    public String activationFunctionTipText() {
        return "Activation Function that is used by PTR. (Value should be between 0 - 2, 0 = SIGN, 1 = STEP, 2 = SIGMOID).";
    }

    public String thresholdTipText() {
        return "Learning rate that is used by the algorithm. Default = 0";
    }

    public String learningRateTipText() {
        return "Learning rate that is used by the algorithm. Default = 0.1";
    }

    public void setLearningRate(float a) {
        learningRate = a;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setHiddenPerceptrons(int a) {
        hiddenPerceptrons = a;
    }

    public int getHiddenPerceptrons() {
        return hiddenPerceptrons;
    }


    public void setMaxIteration(int a) {
        maxIteration = a;
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    // OPSI UNTUK KUSTOMISASI
    public void setOptions(String[] options) throws Exception {
        String learningString = Utils.getOption('L', options);
        if (learningString.length() != 0) {
            learningRate = new Float(learningString).floatValue();
        } else {
            learningRate = 0.1f;
        }

        String activationString = Utils.getOption('H', options);
        if (activationString.length() != 0) {
            hiddenPerceptrons = new Integer(activationString).intValue();
        } else {
            hiddenPerceptrons = 1;
        }

        String maxiterString = Utils.getOption('M', options);
        if (maxiterString.length() != 0) {
            maxIteration = new Integer(maxiterString).intValue();
        } else {
            maxIteration = 30;
        }


        Utils.checkForRemainingOptions(options);
    }


    public String [] getOptions() {
        String [] options = new String [6];
        int current = 0;
        options[current++] = "-L"; options[current++] = "" + learningRate;
        options[current++] = "-H"; options[current++] = "" + hiddenPerceptrons;
        options[current++] = "-M"; options[current++] = "" + maxIteration;

        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }


    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        //result.enable(Capability.DATE_ATTRIBUTES);
        //result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        //result.enable(Capability.DATE_CLASS);
        //result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }

    public static void main(String [] argv) {
        runClassifier(new MLP(), argv);
    }


    public MLP() {

    }

    public String toString() {
        return "OUTPUT MODEL";
    }



    // Mengembalikan informasi mengenai classifier ini

    public String globalInfo() {
        return "Perceptron Training Rule";
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1 $");
    }
}
