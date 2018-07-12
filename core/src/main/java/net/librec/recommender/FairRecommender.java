package net.librec.recommender;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.UserFeatureAppender;
import net.librec.data.convertor.appender.ItemFeatureAppender;
import net.librec.math.algorithm.Maths;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

public abstract class FairRecommender extends MatrixFactorizationRecommender {
    /**
     * userGroupMatrix: only column 0 is used. Represent (as int) the group that user is part of
     */
    protected SparseMatrix m_userGroupMatrix;

    /**
     * itemGroupMatrix: only column 0 is used. Represent (as int) the group that user is part of
     */
    protected SparseMatrix m_itemGroupMatrix;

    /**
     * isUserFair: True if the recommender implements user-side fairness (makes use of protected / unprotected group
     * of users.
     * isItemFair: the same for items.
     */
    protected boolean m_isUserFair;
    protected boolean m_isItemFair;


    @Override
    public void setup() throws LibrecException {
        super.setup();

        m_isUserFair = conf.getBoolean("rec.fair.isuserfair", false);
        m_isItemFair = conf.getBoolean("rec.fair.isitemfair", false);

        if (m_isUserFair) {
            m_userGroupMatrix = ((UserFeatureAppender) getDataModel().getDataAppender()).getUserFeatureAppender();
        }

        if (m_isItemFair) {
            m_itemGroupMatrix = ((ItemFeatureAppender) getDataModel().getDataAppender()).getItemFeatureAppender();
        }
    }

    protected int getUserGroup(int uiid) {
        return (int)m_userGroupMatrix.get(uiid, 0);
    }

    protected SparseMatrix getGroupMatrix() {
        return m_userGroupMatrix;
    }

    protected int getItemGroup(int iiid) {
        return (int)m_itemGroupMatrix.get(iiid, 0);
    }

    protected SparseMatrix getItemMatrix() {
        return m_userGroupMatrix;
    }

}

