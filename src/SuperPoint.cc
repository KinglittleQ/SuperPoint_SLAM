#include "SuperPoint.h"


namespace ORB_SLAM2
{

const int c1 = 64;
const int c2 = 64;
const int c3 = 128;
const int c4 = 128;
const int c5 = 256;
const int d1 = 256;



SuperPoint::SuperPoint()
      : conv1a(torch::nn::Conv2dOptions( 1, c1, 3).stride(1).padding(1)),
        conv1b(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

        conv2a(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
        conv2b(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

        conv3a(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
        conv3b(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

        conv4a(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
        conv4b(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

        convPa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
        convPb(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)),

        convDa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
        convDb(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0))
        
  {
    register_module("conv1a", conv1a);
    register_module("conv1b", conv1b);

    register_module("conv2a", conv2a);
    register_module("conv2b", conv2b);

    register_module("conv3a", conv3a);
    register_module("conv3b", conv3b);

    register_module("conv4a", conv4a);
    register_module("conv4b", conv4b);

    register_module("convPa", convPa);
    register_module("convPb", convPb);

    register_module("convDa", convDa);
    register_module("convDb", convDb);
  }


std::vector<torch::Tensor> SuperPoint::forward(torch::Tensor x) {

    x = torch::relu(conv1a->forward(x));
    x = torch::relu(conv1b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv2a->forward(x));
    x = torch::relu(conv2b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv3a->forward(x));
    x = torch::relu(conv3b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv4a->forward(x));
    x = torch::relu(conv4b->forward(x));

    auto cPa = torch::relu(convPa->forward(x));
    auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]

    auto cDa = torch::relu(convDa->forward(x));
    auto desc = convDb->forward(cDa);  // [B, d1, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    semi = torch::softmax(semi, 1);
    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]


    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]


    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
  }

void NMS(cv::Mat det, cv::Mat conf, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height);
void NMS2(std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint>& pts,
            int border, int dist_thresh, int img_width, int img_height);

cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint> &keypoints, double threshold, bool nms, bool cuda)
{
    auto x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
    x = x.to(torch::kFloat) / 255;

    bool use_cuda = cuda && torch::cuda::is_available();
    torch::DeviceType device_type;
    if (use_cuda)
        device_type = torch::kCUDA;
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);

    model->to(device);
    x = x.set_requires_grad(false);
    auto out = model->forward(x.to(device));
    auto prob = out[0].squeeze(0);  // [H, W]
    auto desc = out[1];             // [1, 256, H/8, W/8]

    auto kpts = (prob > threshold);

    kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)
    auto fkpts = kpts.to(torch::kFloat);
    auto grid = torch::zeros({1, 1, kpts.size(0), 2}).to(device);  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / prob.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / prob.size(0) - 1;  // y

    desc = torch::grid_sampler(desc, grid, 0, 0);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints]

    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]

    if (use_cuda)
        desc = desc.to(torch::kCPU);

    cv::Mat descriptors_no_nms(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data<float>());
    
    std::vector<cv::KeyPoint> keypoints_no_nms;
    for (int i = 0; i < kpts.size(0); i++) {
        float response = prob[kpts[i][0]][kpts[i][1]].item<float>();
        keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
    }

    if (nms) {
        cv::Mat kpt_mat(keypoints_no_nms.size(), 2, CV_32F);
        cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
        for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
            int x = keypoints_no_nms[i].pt.x;
            int y = keypoints_no_nms[i].pt.y;
            kpt_mat.at<float>(i, 0) = (float)keypoints_no_nms[i].pt.x;
            kpt_mat.at<float>(i, 1) = (float)keypoints_no_nms[i].pt.y;

            conf.at<float>(i, 0) = prob[y][x].item<float>();
        }

        cv::Mat descriptors;

        int border = 8;
        int dist_thresh = 4;
        int height = img.rows;
        int width = img.cols;


        NMS(kpt_mat, conf, descriptors_no_nms, keypoints, descriptors, border, dist_thresh, width, height);

        return descriptors;
    }
    else {
        keypoints = keypoints_no_nms;
        return descriptors_no_nms.clone();
    }

    // return descriptors.clone();
}


SPDetector::SPDetector(std::shared_ptr<SuperPoint> _model) : model(_model) 
{
}

void SPDetector::detect(cv::Mat &img, bool cuda)
{
    auto x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
    x = x.to(torch::kFloat) / 255;

    bool use_cuda = cuda && torch::cuda::is_available();
    torch::DeviceType device_type;
    if (use_cuda)
        device_type = torch::kCUDA;
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);

    model->to(device);
    x = x.set_requires_grad(false);
    auto out = model->forward(x.to(device));

    mProb = out[0].squeeze(0);  // [H, W]
    mDesc = out[1];             // [1, 256, H/8, W/8]

}


void SPDetector::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms)
{
    auto prob = mProb.slice(0, iniY, maxY).slice(1, iniX, maxX);  // [h, w]
    auto kpts = (prob > threshold);
    kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)

    std::vector<cv::KeyPoint> keypoints_no_nms;
    for (int i = 0; i < kpts.size(0); i++) {
        float response = prob[kpts[i][0]][kpts[i][1]].item<float>();
        keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
    }

    if (nms) {
        cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
        for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
            int x = keypoints_no_nms[i].pt.x;
            int y = keypoints_no_nms[i].pt.y;
            conf.at<float>(i, 0) = prob[y][x].item<float>();
        }

        // cv::Mat descriptors;

        int border = 0;
        int dist_thresh = 4;
        int height = maxY - iniY;
        int width = maxX - iniX;

        NMS2(keypoints_no_nms, conf, keypoints, border, dist_thresh, width, height);
    }
    else {
        keypoints = keypoints_no_nms;
    }
}


void SPDetector::computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);  // [n_keypoints, 2]  (y, x)

    for (size_t i = 0; i < keypoints.size(); i++) {
        kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
        kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
    }

    auto fkpts = torch::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);

    auto grid = torch::zeros({1, 1, fkpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y

    auto desc = torch::grid_sampler(mDesc, grid, 0, 0);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints]

    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);

    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data<float>());

    descriptors = desc_mat.clone();
}


void NMS2(std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint>& pts,
            int border, int dist_thresh, int img_width, int img_height)
{

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.size(); i++){

        int u = (int) det[i].pt.x;
        int v = (int) det[i].pt.y;

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }
    
    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                if ( confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu) )
                    grid.at<char>(vv + k, uu + j) = 0;
                
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                cv::Point2f p = pts_raw[select_ind];
                float response = conf.at<float>(select_ind, 0);
                pts.push_back(cv::KeyPoint(p, 8.0f, -1, response));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }
    
    // descriptors.create(select_indice.size(), 256, CV_32F);

    // for (int i=0; i<select_indice.size(); i++)
    // {
    //     for (int j=0; j < 256; j++)
    //     {
    //         descriptors.at<float>(i, j) = desc.at<float>(select_indice[i], j);
    //     }
    // }
}

void NMS(cv::Mat det, cv::Mat conf, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height)
{

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.rows; i++){

        int u = (int) det.at<float>(i, 0);
        int v = (int) det.at<float>(i, 1);
        // float conf = det.at<float>(i, 2);

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }
    
    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                if ( conf.at<float>(vv + k, uu + j) < conf.at<float>(vv, uu) )
                    grid.at<char>(vv + k, uu + j) = 0;
                
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                pts.push_back(cv::KeyPoint(pts_raw[select_ind], 1.0f));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }
    
    descriptors.create(select_indice.size(), 256, CV_32F);

    for (int i=0; i<select_indice.size(); i++)
    {
        for (int j=0; j < 256; j++)
        {
            descriptors.at<float>(i, j) = desc.at<float>(select_indice[i], j);
        }
    }
}

} //namespace ORB_SLAM