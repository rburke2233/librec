package net.librec.data.convertor.appender;

import com.google.common.collect.*;
import net.librec.conf.Configuration;
import net.librec.conf.Configured;
import net.librec.data.DataAppender;
import net.librec.math.structure.SparseMatrix;

import java.io.IOException;

/**
 * A <tt>UserItemFeatureAppender</tt> is a class to process and store user feature
 * data and item feature data. Delegates to UserFeatureAppender and ItemFeatureAppender components
 *
 * Configuration notes:
 * See UserFeatureAppender and ItemFeatureAppender
 *
 * @author RBurke
 */
public class UserItemFeatureAppender extends Configured implements DataAppender {

    protected UserFeatureAppender m_userFeatureApp;
    protected ItemFeatureAppender m_itemFeatureApp;

    /**
     * Initializes a newly created {@code UserFeatureAppender} object with null configuration.
     */
    public UserItemFeatureAppender() {
        this(null);
        m_userFeatureApp = new UserFeatureAppender();
        m_itemFeatureApp = new ItemFeatureAppender();
    }

    /**
     * Initializes a newly created {@code UserFeatureAppender} object with a
     * {@code Configuration} object
     *
     * @param conf  {@code Configuration} object for construction
     */
    public UserItemFeatureAppender(Configuration conf) {
        this.conf = conf;
        m_userFeatureApp = new UserFeatureAppender(conf);
        m_itemFeatureApp = new ItemFeatureAppender(conf);
    }

    /**
     * Process appender data.
     *
     * @throws IOException if I/O error occurs during processing
     */
    @Override
    public void processData() throws IOException {
        m_userFeatureApp.processData();
        m_itemFeatureApp.processData();
    }

    /**
     * Get user appender.
     *
     * @return the {@code SparseMatrix} object built by the user feature data.
     */
    public SparseMatrix getUserFeatureMatrix() {
        return m_userFeatureApp.getUserFeatureMatrix();
    }

    public int getUserFeature(String user, int feature) {
        return m_userFeatureApp.getUserFeature(user, feature);
     }

    public int getUserFeature(int userid, int feature) {
        return m_userFeatureApp.getUserFeature(userid, feature);
    }

    /**
     * Set user mapping data.
     *
     * @param userMappingData
     *            user {raw id, inner id} map
     */
    @Override
    public void setUserMappingData(BiMap<String, Integer> userMappingData) {
        m_userFeatureApp.setUserMappingData(userMappingData);
    }

    public SparseMatrix getItemFeatureMatrix() {
        return m_itemFeatureApp.getItemFeatureMatrix();
    }

    public int getItemFeature(String item, int feature) {
        return m_itemFeatureApp.getItemFeature(item, feature);
    }

    public int getItemFeature(int itemid, int feature) {
        return m_itemFeatureApp.getItemFeature(itemid, feature);
    }

    /**
     * Set item mapping data.
     *
     */
    @Override
    public void setItemMappingData(BiMap<String, Integer> itemMappingData) {
        m_itemFeatureApp.setItemMappingData(itemMappingData);
    }
}

