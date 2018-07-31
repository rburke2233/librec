/**
 * Copyright (C) 2016 LibRec
 *
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package net.librec.data.convertor.appender;

import net.librec.BaseTestCase;
import net.librec.common.LibrecException;
import net.librec.data.convertor.TextDataConvertor;
import net.librec.util.DriverClassUtil;
import net.librec.util.ReflectionUtil;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertTrue;

/**
 * User Feature Feature Test Case corresponds to UserFeatureAppender
 * {@link UserFeatureAppender}
 *
 * @author Robin Burke
 */
public class UserFeatureAppenderTestCase extends BaseTestCase {

    @Before
    public void setUp() throws Exception {
        super.setUp();
        conf.set("data.appender.userfeature.class", "net.librec.data.convertor.appender.UserFeatureAppender");
        conf.set("dfs.data.dir", "librec/data/movielens/ml-100k/");
        conf.set("data.input.path", "ratings.txt");

    }

    /**
     * Test the function of read file.
     *
     * @throws IOException
     * @throws ClassNotFoundException 
     */
    @Test
    public void testReadFile() throws IOException, LibrecException, ClassNotFoundException {
        String inputPath = conf.get("dfs.data.dir") + "/" + conf.get("data.input.path");
        TextDataConvertor textDataConvertor = new TextDataConvertor(inputPath);
        textDataConvertor.processData();
        conf.set("data.userfeature.path", "../../test/feature-append/user-features-test.txt");
        String clName = conf.get("data.appender.userfeature.class");
        //Class<?> cl = DriverClassUtil.getClass(clName);
        UserFeatureAppender dataFeature = (UserFeatureAppender) ReflectionUtil.newInstance(Class.forName(clName), conf);
        dataFeature.setUserMappingData(textDataConvertor.getUserIds());
        dataFeature.processData();

        assertTrue(dataFeature.getUserFeature(dataFeature.m_userIds.get("1"), 0) == 1);
        assertTrue(dataFeature.getUserFeature(dataFeature.m_userIds.get("1"), 1) == 0);
        assertTrue(dataFeature.getUserFeature(dataFeature.m_userIds.get("510"), 0) == 0);
        assertTrue(dataFeature.getUserFeature(dataFeature.m_userIds.get("510"), 1) == 1);

        assertTrue(dataFeature.getUserFeatureMatrix().numRows() <= textDataConvertor.getUserIds().size());
    }
}