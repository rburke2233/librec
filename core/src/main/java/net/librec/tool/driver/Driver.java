package net.librec.tool.driver;
import net.librec.conf.Configuration;
import net.librec.job.RecommenderJob;
import net.librec.math.algorithm.Randoms;

import java.io.FileInputStream;
import java.util.Properties;

/**
 * Created by Himan on 12/5/2016.
 */
public class Driver {
    // test git
    // Change this to load a different configuration file.
    public static String CONFIG_FILE =
            "conf/librec.properties";

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String confFilePath = CONFIG_FILE;
        Properties prop = new Properties();
        prop.load(new FileInputStream(confFilePath));
        for (String name : prop.stringPropertyNames()) {
            conf.set(name, prop.getProperty(name));
        }
        System.out.println("Start");
        Randoms.seed(20161205);
        RecommenderJob job = new RecommenderJob(conf);
        job.runJob();
        System.out.print("Finished");
    }
}
