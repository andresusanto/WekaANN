/*
 *    MLP.java
 *    Copyright (C) 2015 by Andre Susanto, Adhika Sigit, Michael Alexander
 *    
 *    Implementation of Multi Layer Perceptron Classifier Algorithm
 */

package weka.classifiers.ann;

import weka.classifiers.Classifier;
import weka.classifiers.functions.neural.LinearUnit;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.neural.NeuralNode;
import weka.classifiers.functions.neural.SigmoidUnit;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;



public class MLP extends Classifier implements OptionHandler, WeightedInstancesHandler {
  
  private static final long serialVersionUID = -5990607817048210779L;

  /**
   * Main method for testing this class.
   *
   * @param argv should contain command line options (see setOptions)
   */
   
  public static void main(String [] argv) {
    runClassifier(new MLP(), argv);
  }
  

  public MLP() {
	  
  }


  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    
    return result;
  }
  
  
  /**
   * Call this function to build and train a neural network for the training
   * data provided.
   * @param i The training data.
   * @throws Exception if can't build classification properly.
   */
  public void buildClassifier(Instances i) throws Exception {
		
  }

  /**
   * Call this function to predict the class of an instance once a 
   * classification model has been built with the buildClassifier call.
   * @param i The instance to classify.
   * @return A double array filled with the probabilities of each class type.
   * @throws Exception if can't classify instance.
   */
  public double[] distributionForInstance(Instance i) throws Exception {
	return null;
  }
  


  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   
  public Enumeration listOptions() {
    
    Vector newVector = new Vector(14);

    newVector.addElement(new Option(
	      "\tLearning Rate for the backpropagation algorithm.\n"
	      +"\t(Value should be between 0 - 1, Default = 0.3).",
	      "L", 1, "-L <learning rate>"));
    newVector.addElement(new Option(
	      "\tMomentum Rate for the backpropagation algorithm.\n"
	      +"\t(Value should be between 0 - 1, Default = 0.2).",
	      "M", 1, "-M <momentum>"));
    newVector.addElement(new Option(
	      "\tNumber of epochs to train through.\n"
	      +"\t(Default = 500).",
	      "N", 1,"-N <number of epochs>"));
    newVector.addElement(new Option(
	      "\tPercentage size of validation set to use to terminate\n"
	      + "\ttraining (if this is non zero it can pre-empt num of epochs.\n"
	      +"\t(Value should be between 0 - 100, Default = 0).",
	      "V", 1, "-V <percentage size of validation set>"));
    newVector.addElement(new Option(
	      "\tThe value used to seed the random number generator\n"
	      + "\t(Value should be >= 0 and and a long, Default = 0).",
	      "S", 1, "-S <seed>"));
    newVector.addElement(new Option(
	      "\tThe consequetive number of errors allowed for validation\n"
	      + "\ttesting before the netwrok terminates.\n"
	      + "\t(Value should be > 0, Default = 20).",
	      "E", 1, "-E <threshold for number of consequetive errors>"));
    newVector.addElement(new Option(
              "\tGUI will be opened.\n"
	      +"\t(Use this to bring up a GUI).",
	      "G", 0,"-G"));
    newVector.addElement(new Option(
              "\tAutocreation of the network connections will NOT be done.\n"
	      +"\t(This will be ignored if -G is NOT set)",
	      "A", 0,"-A"));
    newVector.addElement(new Option(
              "\tA NominalToBinary filter will NOT automatically be used.\n"
	      +"\t(Set this to not use a NominalToBinary filter).",
	      "B", 0,"-B"));
    newVector.addElement(new Option(
	      "\tThe hidden layers to be created for the network.\n"
	      + "\t(Value should be a list of comma separated Natural \n"
	      + "\tnumbers or the letters 'a' = (attribs + classes) / 2, \n"
	      + "\t'i' = attribs, 'o' = classes, 't' = attribs .+ classes)\n"
	      + "\tfor wildcard values, Default = a).",
	      "H", 1, "-H <comma seperated numbers for nodes on each layer>"));
    newVector.addElement(new Option(
              "\tNormalizing a numeric class will NOT be done.\n"
	      +"\t(Set this to not normalize the class if it's numeric).",
	      "C", 0,"-C"));
    newVector.addElement(new Option(
              "\tNormalizing the attributes will NOT be done.\n"
	      +"\t(Set this to not normalize the attributes).",
	      "I", 0,"-I"));
    newVector.addElement(new Option(
              "\tReseting the network will NOT be allowed.\n"
	      +"\t(Set this to not allow the network to reset).",
	      "R", 0,"-R"));
    newVector.addElement(new Option(
              "\tLearning rate decay will occur.\n"
	      +"\t(Set this to cause the learning rate to decay).",
	      "D", 0,"-D"));
    
    
    return newVector.elements();
  }

  
  // OPSI UNTUK KUSTOMISASI
  public void setOptions(String[] options) throws Exception {
    //the defaults can be found here!!!!
    String learningString = Utils.getOption('L', options);
    if (learningString.length() != 0) {
      setLearningRate((new Double(learningString)).doubleValue());
    } else {
      setLearningRate(0.3);
    }
    String momentumString = Utils.getOption('M', options);
    if (momentumString.length() != 0) {
      setMomentum((new Double(momentumString)).doubleValue());
    } else {
      setMomentum(0.2);
    }
    String epochsString = Utils.getOption('N', options);
    if (epochsString.length() != 0) {
      setTrainingTime(Integer.parseInt(epochsString));
    } else {
      setTrainingTime(500);
    }
    String valSizeString = Utils.getOption('V', options);
    if (valSizeString.length() != 0) {
      setValidationSetSize(Integer.parseInt(valSizeString));
    } else {
      setValidationSetSize(0);
    }
    String seedString = Utils.getOption('S', options);
    if (seedString.length() != 0) {
      setSeed(Integer.parseInt(seedString));
    } else {
      setSeed(0);
    }
    String thresholdString = Utils.getOption('E', options);
    if (thresholdString.length() != 0) {
      setValidationThreshold(Integer.parseInt(thresholdString));
    } else {
      setValidationThreshold(20);
    }
    String hiddenLayers = Utils.getOption('H', options);
    if (hiddenLayers.length() != 0) {
      setHiddenLayers(hiddenLayers);
    } else {
      setHiddenLayers("a");
    }
    if (Utils.getFlag('G', options)) {
      setGUI(true);
    } else {
      setGUI(false);
    } //small note. since the gui is the only option that can change the other
    //options this should be set first to allow the other options to set 
    //properly
    if (Utils.getFlag('A', options)) {
      setAutoBuild(false);
    } else {
      setAutoBuild(true);
    }
    if (Utils.getFlag('B', options)) {
      setNominalToBinaryFilter(false);
    } else {
      setNominalToBinaryFilter(true);
    }
    if (Utils.getFlag('C', options)) {
      setNormalizeNumericClass(false);
    } else {
      setNormalizeNumericClass(true);
    }
    if (Utils.getFlag('I', options)) {
      setNormalizeAttributes(false);
    } else {
      setNormalizeAttributes(true);
    }
    if (Utils.getFlag('R', options)) {
      setReset(false);
    } else {
      setReset(true);
    }
    if (Utils.getFlag('D', options)) {
      setDecay(true);
    } else {
      setDecay(false);
    }
    
    Utils.checkForRemainingOptions(options);
  }
  
  
  public String [] getOptions() {

    String [] options = new String [21];
    int current = 0;
    options[current++] = "-L"; options[current++] = "" + getLearningRate(); 
    options[current++] = "-M"; options[current++] = "" + getMomentum();
    options[current++] = "-N"; options[current++] = "" + getTrainingTime(); 
    options[current++] = "-V"; options[current++] = "" +getValidationSetSize();
    options[current++] = "-S"; options[current++] = "" + getSeed();
    options[current++] = "-E"; options[current++] =""+getValidationThreshold();
    options[current++] = "-H"; options[current++] = getHiddenLayers();
    if (getGUI()) {
      options[current++] = "-G";
    }
    if (!getAutoBuild()) {
      options[current++] = "-A";
    }
    if (!getNominalToBinaryFilter()) {
      options[current++] = "-B";
    }
    if (!getNormalizeNumericClass()) {
      options[current++] = "-C";
    }
    if (!getNormalizeAttributes()) {
      options[current++] = "-I";
    }
    if (!getReset()) {
      options[current++] = "-R";
    }
    if (getDecay()) {
      options[current++] = "-D";
    }

    
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }*/
  
  
  /**
   * @return string describing the model.
   */
  public String toString() {
    return "OUTPUT MODEL";
  }

  
  /**
   * This will return a string describing the classifier.
   * @return The string.
   */
  public String globalInfo() {
    return "Multi Layer Perceptron";
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 10073 $");
  }
}
