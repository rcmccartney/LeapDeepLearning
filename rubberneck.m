%Joint optical flow is a resampled motion from start to end
    %We resample to a fixed number of frames
% ( all frames, all joints, 1)- X location of all joints
% (all frames,  all joints, 2)- Y location of all joints
% (all frames, all joints, 3)- Z location of all joints
 
    imgX_in = JointsOF_deltaXYZ(:,:,1);
    imgY_in = JointsOF_deltaXYZ(:,:,2);
    imgZ_in = JointsOF_deltaXYZ(:,:,3);
    [frames,joints] = size(imgX_in);
    outFrames=40;
    JointDiffOF_X = imresize(imgX_in,[outFrames joints],'method','bicubic','Antialiasing',1);
    JointDiffOF_Y = imresize(imgY_in,[outFrames joints],'method','bicubic','Antialiasing',1);
    JointDiffOF_Z = imresize(imgZ_in,[outFrames joints],'method','bicubic','Antialiasing',1);
 