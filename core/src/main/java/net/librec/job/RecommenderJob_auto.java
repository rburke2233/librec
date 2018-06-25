package net.librec.job;

import com.google.common.collect.BiMap;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.DataSplitter;
import net.librec.data.splitter.KCVDataSplitter;
import net.librec.data.splitter.LOOCVDataSplitter;
import net.librec.eval.Measure;
import net.librec.eval.RecommenderEvaluator;
import net.librec.filter.RecommendedFilter;
import net.librec.math.structure.SparseMatrix;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.item.RecommendedItem;
import net.librec.similarity.RecommenderSimilarity;
import net.librec.util.FileUtil;
import net.librec.util.ReflectionUtil;
import org.apache.commons.lang.StringUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.*;

/*
 @LibRec_Auto
 @Aldo-OG
 */

public class RecommenderJob_auto extends RecommenderJob  {
         /*  Attributes  */
    private Configuration conf;
    private Properties global_conf;
    private DataModel dataModel;
    private Map<String, List<Double>> cvEvalResults;

    // counter for test and train split.  Relevant only for CV but works with all split scenarios
    private Integer counter_train;
    private Integer counter_test;

    // counter for sub-experiment, will keep track of all times runJob() is executed when in memory.
    // Presumes reset when running new job from LibRecAuto
    private Integer counter_experiment;


          /*  Constructor  */
    public RecommenderJob_auto(Configuration conf) {
        super(conf);
        this.conf = conf;
        this.counter_train=0;
        this.counter_test=0;
        load_global_conf();
    }

    @Override
    public void runJob() throws LibrecException, IOException, ClassNotFoundException {
        update_experiment_count();
        copy_conf();
        String modelSplit = conf.get("data.model.splitter");
        switch (modelSplit) {
            case "kcv": {
                int cvNumber = conf.getInt("data.splitter.cv.number", 1);
                cvEvalResults = new HashMap<>();
                for (int i = 1; i <= cvNumber; i++) {
                    LOG.info("Splitter info: the index of " + modelSplit + " splitter times is " + i);
                    conf.set("data.splitter.cv.index", String.valueOf(i));
                    executeRecommenderJob_auto();
                }
                printCVAverageResult();
                break;
            }
            case "loocv": {
                String loocvType = conf.get("data.splitter.loocv");
                if (StringUtils.equals("userdate", loocvType) || StringUtils.equals("itemdate", loocvType)) {
                    executeRecommenderJob_auto();
                } else {
                    cvEvalResults = new HashMap<>();
                    for (int i = 1; i <= conf.getInt("data.splitter.cv.number", 1); i++) {
                        LOG.info("Splitter info: the index of " + modelSplit + " splitter times is " + i);
                        conf.set("data.splitter.cv.index", String.valueOf(i));
                        executeRecommenderJob_auto();
                    }
                    printCVAverageResult();
                }
                break;
            }
            case "testset":{
                executeRecommenderJob_auto();
                break;
            }
            case "givenn": {
                executeRecommenderJob_auto();
                break;
            }
            case "ratio": {
                executeRecommenderJob_auto();
                break;
            }
        }
    }

    private void executeRecommenderJob_auto() throws ClassNotFoundException, LibrecException, IOException {
        generateDataModel();
         /*   Save split data -> make into private method   */
        // Needs to be more robust.  Catch non-assignment in XML
        boolean saveFlag = Boolean.parseBoolean(conf.get("save.raw.data"));
        if (saveFlag){saveData_auto();}
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        generateSimilarity(context);
        // save similarities?
        Recommender recommender = (Recommender) ReflectionUtil.newInstance((Class<Recommender>) getRecommenderClass(), conf);
        recommender.recommend(context);
        executeEvaluator(recommender);
        List<RecommendedItem> recommendedList = recommender.getRecommendedList();
        // save UIR rec list
        recommendedList = filterResult(recommendedList);
        saveResult(recommendedList);
    }

    @Override
    public void saveResult(List<RecommendedItem> recommendedList) throws LibrecException, IOException, ClassNotFoundException {
        if (recommendedList != null && recommendedList.size() > 0) {
            // Make output path relevant to token
            update_result_dir();
            //String algoSimpleName = DriverClassUtil.getDriverName(getRecommenderClass());
            String outputPath = conf.get("dfs.result.dir");
            if (null != dataModel && (dataModel.getDataSplitter() instanceof KCVDataSplitter || dataModel.getDataSplitter() instanceof LOOCVDataSplitter) && null != conf.getInt("data.splitter.cv.index")) {
                outputPath = outputPath + "-" + String.valueOf(conf.getInt("data.splitter.cv.index"));
            }
            LOG.info("Result path is " + outputPath);
            // convert itemList to string
            StringBuilder sb = new StringBuilder();
            for (RecommendedItem recItem : recommendedList) {
                String userId = recItem.getUserId();
                String itemId = recItem.getItemId();
                String value = String.valueOf(recItem.getValue());
                sb.append(userId).append(",").append(itemId).append(",").append(value).append("\n");
            }
            String resultData = sb.toString();
            // save resultData
            try {
                FileUtil.writeString(outputPath, resultData);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /*              ~ LibRec_Auto addition ~              */
    private void saveData_auto() throws LibrecException, IOException, ClassNotFoundException {
        //  Set-Up  //
        SparseMatrix test = genTestMatrix();
        SparseMatrix train = genTrainMatrix();

        int numUsersTest = test.numRows();
        int numUsersTrain = train.numRows();

        BiMap<String, Integer> userMapping = dataModel.getUserMappingData();
        BiMap<String, Integer> itemMapping = dataModel.getItemMappingData();

        BiMap<Integer, String> userMappingInverse = userMapping.inverse();
        BiMap<Integer, String> itemMappingInverse = itemMapping.inverse();


              // TRAIN //
        StringBuilder train_out = new StringBuilder();
        for (int i_uid = 0; i_uid < numUsersTrain; i_uid++) {
            SparseVector row_i = train.row(i_uid);
            String userId = userMappingInverse.get(i_uid);
            for (VectorEntry i: row_i){
                String itemId = itemMappingInverse.get(i.index());
                double rating = i.get();
                train_out.append(userId).append("\t").append(itemId).append("\t").append(rating).append("\n");
            }
        }
        String saveDataTrain = train_out.toString();

        //  Write TrainResultData
        try {
            String outputPathTrain = trainFileNameGen();
            FileUtil.writeString(outputPathTrain, saveDataTrain);
        } catch (Exception e) {
            e.printStackTrace();
        }

          // TEST //
        //String outputPathTest = testFileNameGen();
        StringBuilder test_out = new StringBuilder();
        for (int i_uid = 0; i_uid < numUsersTest; i_uid++) {
            SparseVector row_i = test.row(i_uid);
            String userId = userMappingInverse.get(i_uid);
            for (VectorEntry i: row_i){
                String itemId = itemMappingInverse.get(i.index());
                double rating = i.get();
                test_out.append(userId).append("\t").append(itemId).append("\t").append(rating).append("\n");
            }
        }
        String saveDataTest = test_out.toString();

        // Write TestResultData
        try {
            String outputPathTest = testFileNameGen();
            FileUtil.writeString(outputPathTest, saveDataTest);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

          /* Aux functions */

    // Next two functions use excessive reading and writing
    private void load_global_conf(){
        // Init
        Properties global_conf = new Properties();
        InputStream input; // Global conf keeps track of experiment counter...
        try {
            // load a properties file
            input = new FileInputStream("LibRec_Auto/global_conf.properties");
            global_conf.load(input);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.global_conf = global_conf;
    }

    private void update_experiment_count(){
        // This only updates the experiment count and removes any other properties...
        Integer c =  Integer.valueOf(this.global_conf.getProperty("experiment.count"));
        c += 1;
        Properties temp = new Properties();
        temp.setProperty("experiment.count", c.toString());
        try {
            temp.store(new FileOutputStream("LibRec_Auto/global_conf.properties"), null);
        } catch (IOException e) {
            e.printStackTrace();
        }
        load_global_conf();
    }

    private SparseMatrix genTrainMatrix(){ return genMatrixAux(true);}
    private SparseMatrix genTestMatrix(){ return genMatrixAux(false);}

    private SparseMatrix genMatrixAux(boolean i){
        SparseMatrix ret;
        if(i){ ret = dataModel.getDataSplitter().getTrainData(); }
        else { ret = dataModel.getDataSplitter().getTestData(); }
        return ret;
    }

    private String trainFileNameGen() throws IOException, ClassNotFoundException { return fileNameGenAux(true); }
    private String testFileNameGen() throws IOException, ClassNotFoundException { return fileNameGenAux(false); }

    // Generate files for config_>config.prop, data used, etc
    private String fileNameGenAux(Boolean flag) throws IOException, ClassNotFoundException {
        String ret;
        if (flag) {
            // Save split ratio value?
            this.counter_train+=1;
            ret = "LibRec_Auto/Experiments/experiment0" + this.global_conf.getProperty("experiment.count") +
                    "/split/train0"+ this.counter_train.toString();
        }
        else{ this.counter_test+=1;
                ret = "LibRec_Auto/Experiments/experiment0" + this.global_conf.getProperty("experiment.count") +
                        "/split/test0"+ this.counter_test.toString();
        }
        return ret;
    }

    private void update_result_dir(){
        String path = "LibRec_Auto/Experiments/experiment0" + this.global_conf.getProperty("experiment.count") +
                "/results/out0"+ this.counter_train.toString();

        this.conf.set("dfs.result.dir", path);
    }

    private void copy_conf(){
        Path source = Paths.get(System.getProperty("user.dir")+"/conf/conf.properties");
        Path dest = Paths.get(System.getProperty("user.dir")+"/"+"LibRec/Experiments/experiment0"+this.counter_experiment);
        try{
            Files.copy(source,dest,StandardCopyOption.REPLACE_EXISTING);
        }
        catch (IOException e){}
    }

    /***
     *
     *
     *
     *
     *
     *                             * * * Direct copy of LibRec private methods * * *
     *
     *
     *
     *
     *
     ***/

    /**
     * Generate data model.
     *
     * @throws ClassNotFoundException
     * @throws IOException
     * @throws LibrecException
     */

    @SuppressWarnings("unchecked")
    // This generates two models in the end if super is called.  Had to use due to private class in super.  Could be wasteful
    private void generateDataModel() throws ClassNotFoundException, IOException, LibrecException {
        if (null == dataModel) {
            dataModel = ReflectionUtil.newInstance((Class<DataModel>) this.getDataModelClass(), conf);
        }
        dataModel.buildDataModel();
    }

    /**
     * Generate similarity.
     *
     * @param context recommender context
     */
    private void generateSimilarity(RecommenderContext context) {
        String[] similarityKeys = conf.getStrings("rec.recommender.similarities");
        if (similarityKeys != null && similarityKeys.length > 0) {
            for(int i = 0; i< similarityKeys.length; i++){
                if (getSimilarityClass() != null) {
                    RecommenderSimilarity similarity = (RecommenderSimilarity) ReflectionUtil.newInstance(getSimilarityClass(), conf);
                    conf.set("rec.recommender.similarity.key", similarityKeys[i]);
                    similarity.buildSimilarityMatrix(dataModel);
                    if(i == 0){
                        context.setSimilarity(similarity);
                    }
                    context.addSimilarities(similarityKeys[i], similarity);
                }
            }
        }
    }
    /**
     * Execute evaluator.
     *
     * @param recommender  recommender algorithm
     * @throws LibrecException        if error occurs
     * @throws IOException            if I/O error occurs
     * @throws ClassNotFoundException if class not found error occurs
     */
    private void executeEvaluator(Recommender recommender) throws ClassNotFoundException, IOException, LibrecException {
        if (conf.getBoolean("rec.eval.enable")) {
            String[] evalClassKeys = conf.getStrings("rec.eval.classes");
            if (evalClassKeys!= null && evalClassKeys.length > 0) {// Run the evaluator which is
                // designated.
                for(int classIdx = 0; classIdx < evalClassKeys.length; ++classIdx) {
                    RecommenderEvaluator evaluator = (RecommenderEvaluator) ReflectionUtil.newInstance(getEvaluatorClass(evalClassKeys[classIdx]), null);
                    evaluator.setTopN(conf.getInt("rec.recommender.ranking.topn", 10));
                    double evalValue = recommender.evaluate(evaluator);
                    LOG.info("Evaluator info:" + evaluator.getClass().getSimpleName() + " is " + evalValue);
                    collectCVResults(evaluator.getClass().getSimpleName(), evalValue);
                }
            } else {// Run all evaluators
                Map<Measure.MeasureValue, Double> evalValueMap = recommender.evaluateMap();
                if (evalValueMap != null && evalValueMap.size() > 0) {
                    for (Map.Entry<Measure.MeasureValue, Double> entry : evalValueMap.entrySet()) {
                        String evalName = null;
                        if (entry != null && entry.getKey() != null) {
                            if (entry.getKey().getTopN() != null && entry.getKey().getTopN() > 0) {
                                LOG.info("Evaluator value:" + entry.getKey().getMeasure() + " top " + entry.getKey().getTopN() + " is " + entry.getValue());
                                evalName = entry.getKey().getMeasure() + " top " + entry.getKey().getTopN();
                            } else {
                                LOG.info("Evaluator value:" + entry.getKey().getMeasure() + " is " + entry.getValue());
                                evalName = entry.getKey().getMeasure() + "";
                            }
                            if (null != cvEvalResults) {
                                collectCVResults(evalName, entry.getValue());
                            }
                        }
                    }
                }
            }
        }
    }
    /**
     * Collect the evaluate results when using cross validation.
     *
     * @param evalName   name of the evaluator
     * @param evalValue  value of the evaluate result
     */
    private void collectCVResults(String evalName, Double evalValue) {
        DataSplitter splitter = dataModel.getDataSplitter();
        if (splitter != null && (splitter instanceof KCVDataSplitter || splitter instanceof LOOCVDataSplitter)) {
            if (cvEvalResults.containsKey(evalName)) {
                cvEvalResults.get(evalName).add(evalValue);
            } else {
                List<Double> newList = new ArrayList<>();
                newList.add(evalValue);
                cvEvalResults.put(evalName, newList);
            }
        }
    }
    /**
     * Filter the results.
     *
     * @param recommendedList  list of recommended items
     * @return recommended List
     * @throws ClassNotFoundException
     * @throws IOException
     */
    private List<RecommendedItem> filterResult(List<RecommendedItem> recommendedList) throws ClassNotFoundException, IOException {
        if (getFilterClass() != null) {
            RecommendedFilter filter = (RecommendedFilter) ReflectionUtil.newInstance(getFilterClass(), null);
            recommendedList = filter.filter(recommendedList);
        }
        return recommendedList;
    }

    private void printCVAverageResult() {
        LOG.info("Average Evaluation Result of Cross Validation:");
        for (Map.Entry<String, List<Double>> entry : cvEvalResults.entrySet()) {
            String evalName = entry.getKey();
            List<Double> evalList = entry.getValue();
            double sum = 0.0;
            for (double value : evalList) {
                sum += value;
            }
            double avgEvalResult = sum / evalList.size();
            LOG.info("Evaluator value:" + evalName + " is " + avgEvalResult);
        }
    }

    // Closing Body of Class here
}
