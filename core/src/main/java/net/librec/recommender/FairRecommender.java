package net.librec.recommender;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.UserFeatureAppender;
import net.librec.data.convertor.appender.ItemFeatureAppender;
import net.librec.math.algorithm.Maths;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

/**
 * Note that because there can only be one data appender per data model object, we cannot have both user & item
 * features loaded at the moment.
 */
public abstract class FairRecommender extends AbstractRecommender {
    /**
     * userGroupMatrix: only column 0 is used. Represent (as int) the group that user is part of
     */
    protected SparseMatrix m_userGroupMatrix;

    /**
     * itemGroupMatrix: only column 0 is used. Represent (as int) the group that user is part of
     */
    protected SparseMatrix m_itemGroupMatrix;


    @Override
    public void setup() throws LibrecException {
        super.setup();

        String userAppenderClass = conf.get("data.appender.userfeature.class", "");
        String itemAppenderClass = conf.get("data.appender.itemfeature.class", "");

        if (userAppenderClass.isEmpty() == false) {
            UserFeatureAppender ufAppend = (UserFeatureAppender) getDataModel().getDataAppender();
            m_userGroupMatrix = ufAppend.getUserFeatureMatrix();
        } else {
            ItemFeatureAppender ufAppend = (ItemFeatureAppender) getDataModel().getDataAppender();
            m_itemGroupMatrix = ufAppend.getItemFeatureMatrix();
        }
    }

    protected boolean isUserGroup(int uiid, int group) {
        return (int)m_userGroupMatrix.get(uiid, group) != 0;
    }

    protected SparseMatrix getGroupMatrix() {
        return m_userGroupMatrix;
    }

    protected boolean isItemGroup(int iiid, int group) {
        return (int)m_itemGroupMatrix.get(iiid, group) != 0;
    }

    protected SparseMatrix getItemMatrix() {
        return m_itemGroupMatrix;
    }

}

