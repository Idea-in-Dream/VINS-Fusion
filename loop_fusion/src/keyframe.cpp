/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "keyframe.h"
// 根据状态剔除部分特征点
template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 新建关键帧，新建关键帧时，
// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
// 加载关键帧
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;

	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}


// 提取滑动窗口内特征点的BRIEF描述子
void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

// 提取fast特征及BRIEF描述子
void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 20; // corner detector response threshold
	if(1)
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}
	extractor(image, keypoints, brief_descriptors);

	// 计算归一化平面上的特征点坐标
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

// 从给定的关键帧中提取BRIEF描述子
void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


// 在一组在一组特征描述符中寻找与给定描述符最匹配的一个
bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
	// 遍历所有旧的描述符
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {
		// 计算它们与目标描述符 window_descriptor 的汉明距离。
        int dis = HammingDis(window_descriptor, descriptors_old[i]);
		// 找到最小距离的描述符
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }

    // 如果距离小于阈值80，则认为找到了匹配
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    }
    else
      return false;
}

// 批量执行特征匹配，将当前帧的特征点与上一帧特征点进行匹配
void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
		// 遍历当前帧的特征描述符 window_brief_descriptors,调用 searchInAera 查找匹配点，
		// 并将匹配结果存储在 matched_2d_old 和 matched_2d_old_norm 中，同时记录匹配状态 status
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}

// 利用RANSAC对匹配点进行几何一致性验证，估计基本矩阵
void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	// 状态全部置为0
	for (int i = 0; i < n; i++)
		status.push_back(0);

    if (n >= 8)
    {	
		// 将当前帧和上一帧的归一化坐标转换为像素坐标
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
		// 使用 cv::findFundamentalMat 函数进行RANSAC，过滤掉不符合几何一致性的匹配点
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

// 利用PnP（Perspective-n-Point）方法估计关键帧之间的相对位姿
void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	// rvec 和 t：PnP求解的旋转向量和平移向量
    cv::Mat r, rvec, t, D, tmp_r;
	// K：相机内参矩阵 D：畸变参数，这里为空（归一化坐标无需畸变校正）
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
	// 初始位姿计算
	// origin_vio_R 和 origin_vio_T：初始位姿;
	// qic 和 tic：IMU到相机的外参（旋转矩阵和平移向量）
    Matrix3d R_inital;
    Vector3d P_inital;
	// 计算相机在世界坐标系下的旋转 R_w_c 和平移 T_w_c
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

	// 将位姿反转为参考关键帧相对当前帧的初始位姿
    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

	//  位姿初始化到OpenCV格式
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

	// PnP求解
    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

    }
	// 匹配状态更新
    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

	// 将PnP求解结果的旋转向量 rvec 转换为旋转矩阵 r，并转换为Eigen格式 R_pnp
    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);
	
	// 将位姿转换到IMU坐标系下
    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}

// 检测当前关键帧与某一历史关键帧之间是否存在闭环连接
bool KeyFrame::findConnection(KeyFrame* old_kf)
{
	TicToc tmp_t;
	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;

	//printf("search by des\n");
	// 通过 searchByBRIEFDes 查找当前帧与旧帧之间的特征匹配
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	//printf("search by des finish\n");

	status.clear();
	/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/

	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	// 使用 PnPRANSAC 估计两帧之间的相对位姿
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
	    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
	}
	// 如果满足一定的闭环条件（如相对位姿的旋转和平移均小于阈值），标记为闭环连接，并保存相对位姿信息。
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
	    //printf("PNP relative\n");
	    //cout << "pnp relative_t " << relative_t.transpose() << endl;
	    //cout << "pnp relative_yaw " << relative_yaw << endl;
	    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	    	//cout << "pnp relative_t " << relative_t.transpose() << endl;
	    	//cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	        return true;
	    }
	}
	//printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
	return false;
}

// 计算两个BRIEF描述符之间的汉明距离
int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}


