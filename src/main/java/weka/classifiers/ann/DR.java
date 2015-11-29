/*
 *    DR.java
 *    Copyright (C) 2015 by Andre Susanto, Adhika Sigit, Michael Alexander
 *    
 *    Implementation of DELTA RULE Classifier Algorithm
 */

package weka.classifiers.ann;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Vector;
import java.util.Enumeration;

public class DR extends Classifier implements OptionHandler, WeightedInstancesHandler {
    ////// DATA YANG BERKAITAN DENGAN MODEL
    private float learningRate = 0.1f;
    private int maxIteration = 30;
    private float weights[];
    private int activationFunction = 0; // 0 = SIGN, 1 = STEP, 2 = SIGMOID
    private int mode = 0; // 0 = BATCH, 1 = INCREMENTAL
    private double stepThreshold = 0;
    ///////////////////////////////////////
    private NominalToBinary nominalToBinary = new NominalToBinary();
    private Normalize normalize = new Normalize();
    //////////////////////////////////////


    // untuk serialisasi
    private static final long serialVersionUID = -5990607817048210779L;


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

        int i = 0; int it = 0; int iterateTo;
        int sumInstances = instances.numInstances();
        int sumAttributes = instances.numAttributes();

        if (sumInstances > 0) {
            float tmp_weights[] = new float[sumAttributes];
            weights = new float[sumAttributes];
            double curError = 1;

            if (mode == 0){
                iterateTo = maxIteration * sumInstances;
            }else{
                iterateTo = maxIteration;
            }

            while (it < iterateTo && curError > 0) {
                double out = classify(instances.instance(i));
                double correction = instances.instance(i).classValue() - out;

                System.out.printf("Iterasi %d, data %d: (TARGET: %f, OUT: %f)\n", it, i, instances.instance(i).classValue(), out);
                System.out.printf("   NEW WEIGHT: ");

                for (int j = 0; j < sumAttributes; j++){
                    if (j != instances.instance(i).classIndex()){
                        if (mode == 1) {
                            weights[j] = weights[j] + (float) (correction * instances.instance(i).value(j) * learningRate);
                            System.out.printf(" %f,", weights[j]);
                        }else{
                            tmp_weights[j] = tmp_weights[j] + (float) (correction * instances.instance(i).value(j) * learningRate);
                        }
                    }
                }
                curError = calculateError(instances);
                System.out.printf("\n   Error: %f\n\n", curError);

                if (i == sumInstances - 1 && mode == 0){
                    System.out.printf("   UPDATE WEIGHT: ");

                    for (int j = 0; j < sumAttributes; j++){
                        if (j != instances.instance(i).classIndex()){
                            weights[j] = weights[j] + tmp_weights[j];
                        }
                    }
                }

                i = (++i) % sumInstances;
                it++;
            }
        }
    }

    private double classify(Instance _instance) throws Exception{
        Instance instance;

        nominalToBinary.input(_instance);
        instance = nominalToBinary.output();
        normalize.input(instance);
        instance = normalize.output();

        int numAttr = instance.numAttributes();
        double sigma = 0;
        for (int i = 0; i < numAttr; i++){
            if (i != instance.classIndex()){
                sigma += weights[i] * instance.value(i);
            }
        }

        return sigma;
    }


    public double classifyInstance(Instance _instance) throws Exception{
        double sigma = classify(_instance);

        switch (activationFunction){
            case 0: // fungsi sign
                if (sigma > 0) return 1;
                else if (sigma < 0) return -1;
                else return 0;

            case 1: // fungsi step
                if (sigma > stepThreshold)
                    return 1;
                else
                    return 0;
            default: // fungsi sigmoid
                return (1/( 1 + Math.pow(Math.E,(-1* sigma))));
        }
    }


    public Enumeration listOptions() {

        Vector newVector = new Vector(14);

        newVector.addElement(new Option(
              "\tActivation Function that is used by DR.\n"
              +"\t(Value should be between 0 - 2, 0 = SIGN, 1 = STEP, 2 = SIGMOID).",
              "F", 1, "-F <activation function>"));
        newVector.addElement(new Option(
              "\tMaximum Iteration that is used by DR.\n"
              +"\t(Default 30).",
              "M", 1, "-M <max iteration>"));
        newVector.addElement(new Option(
              "\tThreshold that is used by step function (if its used).\n"
              +"\t(Default = 0).",
              "T", 1,"-T <step threshold>"));
        newVector.addElement(new Option(
                "\tLearning Rate.\n"
                +"\t(Default = 0.1).",
                "L", 1,"-L <learning rate>"));
        newVector.addElement(new Option(
                "\tUpdate option (INCREMENTAL or BATCH).\n"
                        +"\t(Default = BATCH, 0 = BATCH, 1 = INCREMENTAL).",
                "O", 1,"-O <update option>"));
        return newVector.elements();
    }

    public String maxIterationTipText() {
        return "Maximum Iteration that is used by DR. Default = 30";
    }

    public String activationFunctionTipText() {
        return "Activation Function that is used by DR. (Value should be between 0 - 2, 0 = SIGN, 1 = STEP, 2 = SIGMOID).";
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

    public void setStepThreshold(double a) {
        stepThreshold = a;
    }

    public double getStepThreshold() {
        return stepThreshold;
    }

    public void setActivationFunction(int a) {
        activationFunction = a;
    }

    public int getActivationFunction() {
        return activationFunction;
    }

    public void setMaxIteration(int a) {
        maxIteration = a;
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    public int getUpdateOption(){ return  mode; }

    public void setUpdateOption(int a) { mode = a; }

    // OPSI UNTUK KUSTOMISASI
    public void setOptions(String[] options) throws Exception {
        String learningString = Utils.getOption('L', options);
        if (learningString.length() != 0) {
            learningRate = new Float(learningString).floatValue();
        } else {
            learningRate = 0.1f;
        }

        String activationString = Utils.getOption('F', options);
        if (activationString.length() != 0) {
            activationFunction = new Integer(activationString).intValue();
        } else {
            activationFunction = 0;
        }

        String maxiterString = Utils.getOption('M', options);
        if (maxiterString.length() != 0) {
            maxIteration = new Integer(maxiterString).intValue();
        } else {
            maxIteration = 30;
        }


        String thresholdString = Utils.getOption('T', options);
        if (thresholdString.length() != 0) {
            stepThreshold = new Integer(thresholdString).intValue();
        } else {
            stepThreshold = 0;
        }

        String updateString = Utils.getOption('O', options);
        if (updateString.length() != 0) {
            mode = new Integer(updateString).intValue();
        } else {
            mode = 0;
        }

        Utils.checkForRemainingOptions(options);
    }

  
    public String [] getOptions() {
        String [] options = new String [10];
        int current = 0;
        options[current++] = "-L"; options[current++] = "" + learningRate;
        options[current++] = "-F"; options[current++] = "" + activationFunction;
        options[current++] = "-M"; options[current++] = "" + maxIteration;
        options[current++] = "-T"; options[current++] = "" + stepThreshold;
        options[current++] = "-O"; options[current++] = "" + mode;

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
        runClassifier(new DR(), argv);
    }


    public DR() {

    }
  
	public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("DELTA RULE MODEL\n");
        builder.append("----------------\n");

        for (int i = 0 ; i < weights.length; i++){
            builder.append("   W[");
            builder.append(i);
            builder.append("] = ");
            builder.append(weights[i]);
            builder.append("\n");
        }

		return builder.toString();
	}

  

	// Mengembalikan informasi mengenai classifier ini
	
	public String globalInfo() {
		return "Perceptron Training Rule";
	}

	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1 $");
	}
}
