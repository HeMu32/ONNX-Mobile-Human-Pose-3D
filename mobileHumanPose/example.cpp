void YOLO::sigmoid(Mat* out, int length)
{
    float* pdata = (float*)(out->data);
    int i = 0; 
    for (i = 0; i < length; i++)
    {
        pdata[i] = 1.0 / (1 + expf(-pdata[i]));
    }
}

void YOLO::detect(Mat& frame)
{
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
    
    /////generate proposals
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
    int n = 0, q = 0, i = 0, j = 0, nout = this->classes.size() + 5, c = 0;
    for (n = 0; n < 3; n++)   //三层特征图
    {
        int num_grid_x = (int)(this->inpWidth / this->stride[n]);
        int num_grid_y = (int)(this->inpHeight / this->stride[n]);
        int area = num_grid_x * num_grid_y;
        this->sigmoid(&outs[n], 3 * nout * area);  //计算每个值的sigmoid值
        for (q = 0; q < 3; q++)    ///anchor数
        {
            const float anchor_w = this->anchors[n][q * 2];
            const float anchor_h = this->anchors[n][q * 2 + 1];
            float* pdata = (float*)outs[n].data + q * nout * area;   //偏移=anchor索引 * 85 * 特征图面积
            for (i = 0; i < num_grid_y; i++)  
            {
                for (j = 0; j < num_grid_x; j++)
                {
                                        //每个anchor在整个特征图上的每个anchor的值
                    float box_score = pdata[4 * area + i * num_grid_x + j];  
                    if (box_score > this->objThreshold)
                    {
                        float max_class_socre = 0, class_socre = 0;
                        int max_class_id = 0;
                        for (c = 0; c < this->classes.size(); c++) //// get max socre
                        {
                            class_socre = pdata[(c + 5) * area + i * num_grid_x + j];
                            if (class_socre > max_class_socre)
                            {
                                max_class_socre = class_socre;
                                max_class_id = c;
                            }
                        }
                        
                        if (max_class_socre > this->confThreshold)
                        {
                            float cx = (pdata[i * num_grid_x + j] * 2.f - 0.5f + j) * this->stride[n];  ///cx
                            float cy = (pdata[area + i * num_grid_x + j] * 2.f - 0.5f + i) * this->stride[n];   ///cy
                            float w = powf(pdata[2 * area + i * num_grid_x + j] * 2.f, 2.f) * anchor_w;   ///w
                            float h = powf(pdata[3 * area + i * num_grid_x + j] * 2.f, 2.f) * anchor_h;  ///h
                            
                            int left = (cx - 0.5*w)*ratiow;
                            int top = (cy - 0.5*h)*ratioh;   ///坐标还原到原图上

                            classIds.push_back(max_class_id);
                            confidences.push_back(max_class_socre);
                            boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
                        }    
                    }    
                }
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}