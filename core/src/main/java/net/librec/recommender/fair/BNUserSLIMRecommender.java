package net.librec.recommender.fair;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.FairRecommender;
import net.librec.util.Lists;
import org.apache.commons.lang.math.NumberUtils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * This work is heavily borrowed from
 * Xia Ning and George Karypis, <strong>SLIM: Sparse Linear Methods for Top-N Recommender Systems</strong>, ICDM 2011. <br>
 * except that this method is used for User SLIM instead of item SLIM and
 * The objective function includes user balance as well to achive parity.
 * written by Nasim Sonboli
 */
@ModelData({"isRanking", "slim", "coefficientMatrix", "trainMatrix", "similarityMatrix", "knn"})
public class BNUserSLIMRecommender extends FairRecommender {

    /**
     * the number of iterations
     */
    protected int numIterations;

    /**
     * W in original paper, a sparse matrix of aggregation coefficients
     */
    private DenseMatrix coefficientMatrix;

    /**
     * user's nearest neighbors for kNN > 0
     */
    private Set<Integer>[] userNNs;

    /**
     * regularization parameters for the L1 or L2 term
     */
    private float regL1Norm, regL2Norm;

    /**
     * This parameter sets a threshold for similarity, so we only consider the user pairs that their sim > threshold.
     */
    private float efficiencySimThresh;

    /**
     *This parameter controls the influence of user balance calculation on the overall optimization.
     */
    private float Lambda3;

    /**
     * This vector is a 1 x M vector, and M is the number of users,
     * this vector is filled with either 1 or -1,
     * If a user belongs to the protected group it is +1, otherwise it is -1
     */
    private double[] groupMembershipVector;
//    int[] groupMembershipVector = new int[numUsers];

    /**
     * number of nearest neighbors
     */
    protected static int knn;


    /**
     * item similarity matrix
     */
    private SymmMatrix similarityMatrix;

    /**
     * users's nearest neighbors for kNN <=0, i.e., all other items
     */
    private Set<Integer> allUsers;

    /**
     * balance
     */
    private static double balance;

    /**
     * reading the membership file
     */
    private String membershipFilePath;

    /**
     * initialization
     *
     * @throws LibrecException if error occurs
     */
    @Override
    public void setup() throws LibrecException {
        super.setup();
        knn = conf.getInt("rec.neighbors.knn.number", 50);
        numIterations = conf.getInt("rec.iterator.maximum");
        regL1Norm = conf.getFloat("rec.slim.regularization.l1", 1.0f);
        regL2Norm = conf.getFloat("rec.slim.regularization.l2", 1.0f);
        Lambda3 = conf.getFloat("rec.bnslim.regularization.l3", 1.0f);
        efficiencySimThresh = conf.getFloat("rec.bnslim.minsimilarity",0.0f);
        // Set it in configuration file


        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
        System.out.println("Done fetching the similarity Matrix...");

        //We have 6040 users
//        System.out.println();
//        for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
//            SparseVector similarityVector = similarityMatrix.row(userIdx);
//        }

        coefficientMatrix = new DenseMatrix(numUsers, numUsers);
        // initial guesses: make smaller guesses (e.g., W.init(0.01)) to speed up training
        coefficientMatrix.init();

        for(int userIdx = 0; userIdx < this.numUsers; ++userIdx) {
            this.coefficientMatrix.set(userIdx, userIdx, 0.0d);
        } //iterate through all of the users , initialize
        System.out.println("Done creating initializing the coefficient matrix...");

        createUserNNs();
        System.out.println("Done creating the nearest neighbor matrix...");


        groupMembershipVector = new double[numUsers];
        //fill in the membership vector by membership numbers (1, -1)
        for (int userIdx = 0; userIdx < numUsers; userIdx++) {

            // Must be group 0 or 1, other groups ignored. By convention, group 0 is protected.
            if (isUserGroup(userIdx, 0)) {
                groupMembershipVector[userIdx] = 1;
            } else {
                if (isUserGroup(userIdx, 1)) {
                    groupMembershipVector[userIdx] = -1;
                }
            }
        }
        System.out.println("Done setting up membership vector...!");
    }


    /**
     * train model
     *
     * @throws LibrecException if error occurs
     */
    @Override
    protected void trainModel() throws LibrecException {

        //efficiency: Calculate the correlation between the users and exclude the ones that are lower than a threshold



        // number of iteration cycles
        for (int iter = 1; iter <= numIterations; iter++) {

            loss = 0.0d;
            // each cycle iterates through one coordinate direction
            for (int userIdx = 0; userIdx < numUsers; userIdx++) {
                // find k-nearest neighbors of each user
                // if we have set knn to a number, it is > 0, otherwise all the users are knns of a user
                Set<Integer> nearestNeighborCollection = knn > 0 ? userNNs[userIdx] : allUsers;

                //all the ratings of userIdx for all the items
                double[] itemRatingEntries = new double[numItems];

                Iterator<VectorEntry> itemItr = trainMatrix.colIterator(userIdx); //should we go through all of the items??????
                while (itemItr.hasNext()) {
                    VectorEntry itemRatingEntry = itemItr.next();
                    itemRatingEntries[itemRatingEntry.index()] = itemRatingEntry.get();
                }

                int nnIncluded = 0; //efficiency

                // for each nearest neighbor nearestNeighborItemIdx, update coefficienMatrix by the coordinate
                // descent update rule
                for (Integer nearestNeighborUserIdx : nearestNeighborCollection) { //user nearest neighbors!


                    double sim = similarityMatrix.get(nearestNeighborUserIdx, userIdx); //efficiency
//                    System.out.print(sim);

                    if (nearestNeighborUserIdx != userIdx && sim > efficiencySimThresh ) { //efficiency
                        nnIncluded ++;

                        double gradSum = 0.0d, rateSum = 0.0d, errors = 0.0d, userBalanceSum =0.0d, userBalanceSumSqr =0.0d;

                        //ratings of each user on all the other items
                        Iterator<VectorEntry> nnItemRatingItr = trainMatrix.colIterator(nearestNeighborUserIdx);
                        if (!nnItemRatingItr.hasNext()) {
                            continue;
                        }

                        int nnCount = 0;

                        while (nnItemRatingItr.hasNext()) { // now go through the ratings a user has put on items
                            VectorEntry nnItemVectorEntry = nnItemRatingItr.next();
                            int nnItemIdx = nnItemVectorEntry.index();
                            double nnRating = nnItemVectorEntry.get();
                            double rating = itemRatingEntries[nnItemIdx]; // rating of userIdx on nnItemIdx

                            // Error = Actual rating of user on nnItem - prediction of user on nnItem
//                            double error = rating - predict(userIdx, nnItemIdx, nearestNeighborUserIdx);
                            double error = rating - predict_both(userIdx, nnItemIdx, nearestNeighborUserIdx);

                            // Calculating Sigma(pk . wik)
//                            double userbalance = balancePredictor(userIdx, nnItemIdx, nearestNeighborUserIdx);
                            double userbalance = balance;
                            // we have already updated it.
                            //ui and uk should be excluded, end of story!!!


                            userBalanceSumSqr += userbalance * userbalance; //user balance squared
                            userBalanceSum += userbalance;
                            gradSum += nnRating * error;
                            rateSum += nnRating * nnRating; // sigma r^2

                            errors += error * error;
                            nnCount++;
                        }

                        userBalanceSum /= nnCount; // Doubt: user balance sum (?) why should we divide it by nnCount?
                        userBalanceSumSqr /= nnCount;
                        gradSum /= nnCount;
                        rateSum /= nnCount;
                        errors /= nnCount;


                        double coefficient = coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
                        double nnMembership = groupMembershipVector[userIdx];
                        // Loss function
                        loss += 0.5 * errors + 0.5 * regL2Norm * coefficient * coefficient +
                                regL1Norm * coefficient + 0.5 * Lambda3 * userBalanceSumSqr ;


                        /** Implementing Soft Thresholding => S(beta, Lambda1)+
                         * beta = Sigma(r - Sigma(wr)) + Lambda3 * p * Sigma(wp)
                         * & Sigma(r - Sigma(wr)) = gradSum
                         * & nnMembership = p
                         * & Sigma(wp) = userBalanceSum
                         */
                        double beta = gradSum + (Lambda3 * nnMembership * userBalanceSum) ; //adding user balance to the gradsum
                        double update = 0.0d; //weight

                        if (regL1Norm < Math.abs(beta)) {
                            if (beta > 0) {
                                update = (beta - regL1Norm) / (regL2Norm + rateSum + Lambda3);
                            } else {
                                // One doubt: in this case, wij<0, however, the
                                // paper says wij>=0. How to gaurantee that?
                                update = (beta + regL1Norm) / (regL2Norm + rateSum + Lambda3);
                            }
                        }
                        // We can decide whether to keep negative weights or not, nasim
//                        if (update < 0) {
//                            update = 0;
//                        }
                        coefficientMatrix.set(nearestNeighborUserIdx, userIdx, update); //update the coefficient
                    }

                }
            }

            if (isConverged(iter) && earlyStop) {
                break;
            }
        }
    }


//    /**
//     * predict a specific ranking score for user userIdx on item itemIdx.
//     *
//     * @param userIdx         user index
//     * @param itemIdx         item index
//     * @param excludedUserIdx excluded user index
//     * @return a prediction without the contribution of excluded item
//     */
//    protected double predict(int userIdx, int itemIdx, int excludedUserIdx) {
//        double predictRating = 0;
//        Iterator<VectorEntry> userEntryIterator = trainMatrix.rowIterator(itemIdx);
//        while (userEntryIterator.hasNext()) {
//            VectorEntry userEntry = userEntryIterator.next();
//            int nearestNeighborUserIdx = userEntry.index(); //nn user
//            double nearestNeighborPredictRating = userEntry.get();
//            if (userNNs[userIdx].contains(nearestNeighborUserIdx) && nearestNeighborUserIdx != excludedUserIdx) {
//                predictRating += nearestNeighborPredictRating * coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
//            }
//        }
//
//        return predictRating;
//    }

    /**
     * In this function we'll try to efficiency both calculate the predicted rating for user u and item i, and
     * also user balance.
     * @param userIdx
     * @param itemIdx
     * @param excludedUserIdx
     * @return
     */
    protected double predict_both(int userIdx, int itemIdx, int excludedUserIdx) {

        //efficiency
        double predictRating = 0;
        balance = 0; //testing efficient

        Iterator<VectorEntry> userEntryIterator = trainMatrix.rowIterator(itemIdx);

        while (userEntryIterator.hasNext()) {
            //iterate through the nearest neighbors of a user and calculate the prediction accordingly
            VectorEntry userEntry = userEntryIterator.next();
            int nearestNeighborUserIdx = userEntry.index(); //nn user index
            double nearestNeighborPredictRating = userEntry.get();

            if (userNNs[userIdx].contains(nearestNeighborUserIdx) && nearestNeighborUserIdx != excludedUserIdx) {

                double coeff = coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
                //predictRating += nearestNeighborPredictRating * coefficientMatrix.get(nearestNeighborUserIdx, userIdx);
                //Calculate the prediction
                predictRating += nearestNeighborPredictRating * coeff;
                //calculate the user balance
                //take p vector, multiply by the coefficients of neighbors (dot product)
                balance += groupMembershipVector[nearestNeighborUserIdx] * coeff;
            }
        }
        return predictRating;
    }

    @Override
    protected boolean isConverged(int iter) {
        double delta_loss = lastLoss - loss;
        lastLoss = loss;

        // print out debug info
        if (verbose) {
            String recName = getClass().getSimpleName().toString();
            String info = recName + " iter " + iter + ": loss = " + loss + ", delta_loss = " + delta_loss;
            LOG.info(info);
        }

        return iter > 1 ? delta_loss < 1e-5 : false;
    }


    /**
     * predict a specific ranking score for user userIdx on item itemIdx.
     *
     * @param userIdx user index
     * @param itemIdx item index
     * @return predictive ranking score for user userIdx on item itemIdx
     * @throws LibrecException if error occurs
     */
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
//        create item knn list if not exists,  for local offline model
        if (!(null != userNNs && userNNs.length > 0)) {
            createUserNNs();
        }
//        return predict(userIdx, itemIdx, -1);
        return predict_both(userIdx, itemIdx, -1);
    }


    /**
     * Create user KNN list.
     */
    public void createUserNNs() {
        userNNs = new HashSet[numUsers];

        // find the nearest neighbors for each user based on user similarity
        List<Map.Entry<Integer, Double>> tempUserSimList;
        if (knn > 0) {
            for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
                SparseVector similarityVector = similarityMatrix.row(userIdx);
                if (knn < similarityVector.size()) {
                    tempUserSimList = new ArrayList<>(similarityVector.size() + 1);
                    Iterator<VectorEntry> simItr = similarityVector.iterator();
                    while (simItr.hasNext()) {
                        VectorEntry simVectorEntry = simItr.next();
                        //efficiency : if simVectorEntry.get or the similarity is lower than a threshold, don't include it in the userNNs
                        tempUserSimList.add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get()));
                    }
                    tempUserSimList = Lists.sortListTopK(tempUserSimList, true, knn);
                    userNNs[userIdx] = new HashSet<>((int) (tempUserSimList.size() / 0.5)); // why 0.5?? why not * 2?
                    for (Map.Entry<Integer, Double> tempUserSimEntry : tempUserSimList) {
                        userNNs[userIdx].add(tempUserSimEntry.getKey());
                    }
                } else {
                    userNNs[userIdx] = similarityVector.getIndexSet();
                }
            }
        } else {
            allUsers = new HashSet<>(trainMatrix.rows());
        }
    }
}

