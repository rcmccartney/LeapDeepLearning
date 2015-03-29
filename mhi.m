function [mhi_features] = mhi(avi_video,mhi_options)

%for FFD, use:
%[Ireg,O_trans,Spacing,M,B,F] = image_registration(img1,img2,options);
%Then use F(:,:,1) as the x vector and F(:,:,2) as the y vector input
%Resolution of img1 and img2 should be final resolution of vector fields
%options.MaxRef- set to 1 for fast, and set to 2 (default) to slow

if (0)
avi_video = allxy3BigA_test_avi; 
avi_video = outKinect;
mhi_options.skipFirstNframes=1; %number of frames to skip in beginning of video
mhi_options.skipLastNframes=0; %number of frames to skip at end of video
mhi_options.thetaList = [2 4 8 12 16 20]; %temporal window sizes
mhi_options.gaussianWidth = 3;  %gaussian blur filter width applied to image
mhi_options.gaussianSigma = 0.5; %gaussian blur filter sigma applied to image
mhi_options.noiseThreshold = 4;  %pixels need to differ by more than this
mhi_options.binaryOpeningMinArea = 5;  %areas smaller than this will be ignored
end

[numFrames,FrameHeight,FrameWidth] = size(avi_video);
firstValidFrame = 1+mhi_options.skipFirstNframes;
lastValidFrame = numFrames-mhi_options.skipLastNframes;

index = find(mhi_options.thetaList == -1);
if ~isempty(index)
    allFrameCount = lastValidFrame-firstValidFrame+1; 
    %allFrameCount = numFrames-1; %we will skip first frame
    %allFrameCount = 2*floor(allFrameCount/2);
    %mhi_options.thetaList(index) = allFrameCount;
end



%do initial gaussian blur
h = fspecial('gaussian',mhi_options.gaussianWidth,mhi_options.gaussianSigma);
for i=firstValidFrame:lastValidFrame
    avi_video(i,:,:) = imfilter(avi_video(i,:,:),h);
end
% testface = zeros(FrameHeight,FrameWidth);
% testface(:) = avi_video(3,:,:);
% figure; imshow(uint8(testface)); 

%%
theta = max(mhi_options.thetaList);
if theta == -1
    theta = lastValidFrame-firstValidFrame+1; 
    %theta = numFrames-1;
end
tstart = floor(theta/2)+firstValidFrame;  %start t frame
tend = lastValidFrame - ceil(theta/2)+1;  %end t frame
while tstart > tend
    sprintf('Video has %d frames, first valid frame=%d, last valid frame=%d\n',numFrames,firstValidFrame,lastValidFrame)
    sprintf('Using a window size of %d, not enough frames (%d) in video\n',theta,(lastValidFrame-firstValidFrame+1))
    sprintf('Duping last frame\n\n')
    avi_video = [avi_video ; avi_video(end,:,:)]; %dupe last frame
    lastValidFrame = lastValidFrame+1;
    tend = tend+1;  %end t frame
end
% if tstart > tend
%     sprintf('Video has %d frames, first valid frame=%d, last valid frame=%d\n',numFrames,firstValidFrame,lastValidFrame)
%     error(sprintf('Using a window size of %d, not enough frames () in video\n',theta,(lastValidFrame-firstValidFrame+1)));
% end
skipFrames=4;
framelist = tstart:skipFrames:tend;
if length(framelist) > 20
    framelist = round(linspace(tstart,tend,20));
end
%%
thetanum=1;
for thetanum=1:length(mhi_options.thetaList)
    theta = mhi_options.thetaList(thetanum);
    if theta == -1
        theta = lastValidFrame-firstValidFrame+1; 
        %theta = numFrames -1;
    end
    
%     tstart = round(theta/2)+firstValidFrame;  %start t frame
%     tend = lastValidFrame - round(theta/2)+1;  %end t frame
%     if tstart > tend
%         sprintf('Using a window size of %d, not enough frames () in video\n',theta,(lastValidFrame-firstValidFrame+1));
%         continue
%     end
    
    framet = zeros(FrameHeight,FrameWidth);
    frametp1 = zeros(FrameHeight,FrameWidth);
    dframeOpen = zeros(theta-1,FrameHeight,FrameWidth);
    t=framelist(1); framepairID=1;
    se = strel('disk',round((mhi_options.binaryOpeningMinArea/pi).^0.5));
    tcount=0;
    for toffset = 1:length(framelist)  %center t over all frames in video
        t =framelist(toffset);
        tcount=tcount+1;
        for framepairID=1:(theta-1)  %compute all delta frames centered on t
            firstframeNum = t - floor(theta/2)+ framepairID-1;
            framet(:) = avi_video(firstframeNum,:,:);
            frametp1(:) = avi_video(firstframeNum+1,:,:);
            dframe = (abs(framet-frametp1) > mhi_options.noiseThreshold);
            dframeOpen(framepairID,:,:) = imopen(dframe,se);
            disp(sprintf('Theta: %f, t: %f, firstframeNum: %d',theta,t,firstframeNum));
            if (0 && theta==4) 
                str = sprintf('C:\\Data\\Users\\L623945\\face_database\\GEMEP\\emotion_test\\test\\test_044\\sample_images\\mhi\\mhi_diff_theta%d_offset%d_frames%d_%d.jpg',theta,toffset,firstframeNum,firstframeNum+1);
                imwrite(imopen(dframe,se),str ); 
                imshow(imopen(dframe,se))
                %pause
            end
        end
        
        %motion-history images (MHI) represent how the face is moving  The pixel
        %intensity will represent the temporal history of motion at a
        %point.  More recently moving pixels will be higher.
        mframe = zeros(FrameHeight,FrameWidth);
        for h=1:FrameHeight  
            for w=1:FrameWidth
                for framepairID=1:(theta-1)
                    if dframeOpen(framepairID,h,w) > 0
                        mframe(h,w) = round(framepairID*255/(theta-1)); %scale 0:255
                    end
                end
            end
        end
        if (0 && theta==4) 
            str = sprintf('C:\\Data\\Users\\L623945\\face_database\\GEMEP\\emotion_test\\test\\test_044\\sample_images\\mhi\\mhi_weighted_theta%d_offset%d_frames%d_%d.jpg',theta,toffset,t - round(theta/2),t - round(theta/2)+theta-1);
            imwrite(uint8(mframe),str ); 
            imshow(uint8(mframe))
        end
        if mhi_options.thetaList(thetanum) == -1
            eval(['mhiframe_Theta9999(' num2str(tcount) ',:,:)=mframe;']);
            eval(['mhi_features.mhiframe_Theta9999(' num2str(tcount) ',:,:)=mframe;']);
        else
            eval(['mhiframe_Theta' num2str(theta) '(' num2str(tcount) ',:,:)=mframe;']);   
            eval(['mhi_features.mhiframe_Theta' num2str(theta) '(' num2str(tcount) ',:,:)=mframe;']);   
        end
        
        
        %For each pixel in the dframeOpen sequence, record the longest
        %non-zero occurance of motion...this records the magnitude of the
        %motion...these are magnitude images
        mframe = zeros(FrameHeight,FrameWidth);
        mframeB = zeros(FrameHeight,FrameWidth);
        for h=1:FrameHeight  
            for w=1:FrameWidth
                for framepairID=1:(theta-1)
                    if dframeOpen(framepairID,h,w) > 0
                        mframe(h,w) = mframe(h,w)+1;
                    else
                        if mframe(h,w) > mframeB(h,w)
                            mframeB(h,w) = mframe(h,w);
                        end
                        mframe(h,w) = 0;
                    end
                end
                if mframe(h,w) > mframeB(h,w)
                    mframeB(h,w) = mframe(h,w);
                end
                mframeB(h,w) = round(mframeB(h,w)*255/(theta-1)); %scale 0:255
            end
        end
        if mhi_options.thetaList(thetanum) == -1
            eval(['mframe_Theta9999(' num2str(tcount) ',:,:)=mframeB;']);
            eval(['mhi_features.mframe_Theta9999(' num2str(tcount) ',:,:)=mframeB;']);  
        else
            eval(['mframe_Theta' num2str(theta) '(' num2str(tcount) ',:,:)=mframeB;']);
            eval(['mhi_features.mframe_Theta' num2str(theta) '(' num2str(tcount) ',:,:)=mframeB;']);
        end
    end
end

%% Turn MHI image into motion vectors- new method, under construction
% for each pixel that is not the brightest or darkest in the image, search
% neighborhood for brightest pixel that is connected in a monotone or
% continuously increasing fashion as we go from center outward
clear mhi_features.mhi_mag_frame_Theta* mhi_features.mhi_dir_frame_Theta* 
clear mhi_features.mhi_X_frame_Theta* mhi_features.mhi_Y_frame_Theta*
if mhi_options.thetaList(end) == -1
    eval(['numPredictors = size(mhiframe_Theta9999,1);']);
else
    eval(['numPredictors = size(mhiframe_Theta' num2str(mhi_options.thetaList(end)) ',1);']);
end
testfaceStart = zeros(FrameHeight,FrameWidth);
mhi_dir = zeros(round(FrameHeight/2),round(FrameWidth/2));
mhi_mag = zeros(round(FrameHeight/2),round(FrameWidth/2));
mhi_X = zeros(round(FrameHeight/2),round(FrameWidth/2));
mhi_Y = zeros(round(FrameHeight/2),round(FrameWidth/2));
dirLUT1 = [3*pi/4 pi/2  pi/4 pi 0 5*pi/4 3*pi/2 7*pi/4];
dirLUT2 = [ 3*pi/4  5*pi/8  pi/2   3*pi/8    pi/4; ...
            7*pi/8  3*pi/4  pi/2     pi/4    pi/8; ...
              pi      pi     0        0       0; ...
            9*pi/8  5*pi/4 3*pi/2  7*pi/4 15*pi/8; ...
            5*pi/4 11*pi/8 3*pi/2 13*pi/8  7*pi/4];
neighborlist5 = zeros(7,7);
neighborlist5 = zeros(5,5);
neighborlist3 = zeros(3,3);
mask3 = zeros(3,3);
mask5=zeros(5,5);
i=1; thetanum=1;
for i=1:numPredictors
%for i=3:3
    for thetanum=1:length(mhi_options.thetaList)
    %for thetanum=5:5
        theta = mhi_options.thetaList(thetanum);
        if theta == -1
            if firstValidFrame > 1
                theta = lastValidFrame-firstValidFrame+1; 
            else
               theta = lastValidFrame-firstValidFrame; %we will skip first frame
            end
            %theta = numFrames -1;
        end
        [i theta]
        if mhi_options.thetaList(thetanum) == -1
            %this was method used for ICIP2012
            eval(['testfaceStart(:) = mhiframe_Theta9999(i,:,:);']);
            %this is a new test method based on motion duration
            %eval(['testfaceStart(:) = mframe_Theta9999(i,:,:);']);
        else
            %this was method used for ICIP2012
            eval(['testfaceStart(:) = mhiframe_Theta' num2str(theta) '(i,:,:);']);
            %this is a new test method based on motion duration
            %eval(['testfaceStart(:) = mframe_Theta' num2str(theta) '(i,:,:);']);
        end
        
%         testface(10:30,5:15)=255;
%         testface(30:40,5:25)=255;
%         testface(10:35,25:40)=0;
        figure(1); imshow(uint8(testfaceStart));
        filt = fspecial('gaussian',mhi_options.gaussianWidth*2,mhi_options.gaussianSigma*2);
        testface = imfilter(testfaceStart,filt);
        testface = imresize(testface,0.5);
        figure(2); imshow(uint8(testface));
        [WorkingHeight,WorkingWidth] = size(testface);
        h=4; w=4;
        for h=4:(WorkingHeight-3)  
            for w=4:(WorkingWidth-3)
    %             neighborlist = [testface(h-1,w-1) testface(h-1,w) testface(h-1,w+ 1) ...
    %                 testface(h,w-1)  testface(h,w+ 1) ...
    %                 testface(h+1,w-1) testface(h+1,w) testface(h+1,w+ 1)];
    %             mval = max(neighborlist);
                neighborlist7 = testface(h-3:h+3,w-3:w+3);
                neighborlist5 = testface(h-2:h+2,w-2:w+2);
                neighborlist3 = testface(h-1:h+1,w-1:w+1);
                current = testface(h,w);
                a3mask = neighborlist3 >= current;
                a5mask = neighborlist5 >= current;
                a7mask = neighborlist7 >= current;
                
                sq3 = strel('square',3); %create 3x3 dialation mask
                valid5=zeros(5,5);    %dialate 3x3 to denote valid 5x5 locs
                valid5(2:4,2:4) = a3mask;  
                valid5 = imdilate(valid5,sq3);
                a5mask = a5mask .*valid5; %only keep valid 5x5 locs
                
                valid7=zeros(7,7);    %dialate remaining 5x5 to denote valid 5x5 locs
                valid7(2:6,2:6) = a5mask;  
                valid7 = imdilate(valid7,sq3);
                a7mask = a7mask .*valid7; %only keep valid 5x5 locs
                
                runningX=0; runningY=0;
                for r=1:7
                    for c=1:7
                        allCellX(r,c) = c-4; 
                        allCellY(r,c) = -1*(r-4); 
                        if a7mask(r,c)
                            cellTheta = atan2(allCellY(r,c),allCellX(r,c));
                            allTheta(r,c) = cellTheta;
                            runningX = runningX + neighborlist7(r,c)*cos(cellTheta);
                            runningY = runningY + neighborlist7(r,c)*sin(cellTheta);
                        else
                            allTheta(r,c) = 0;
                        end 
                    end
                end
                FinalDir = atan2(runningY,runningX)*180/pi;
                if FinalDir < 0
                    FinalDir = 360+FinalDir;
                end
                FinalMag = sqrt(runningX.^2 + runningY.^2);
                FinalMag = FinalMag *360 / (255*4*7);  %normalize so max mag = 360
                %[FinalDir FinalMag]
                mhi_mag(h,w) = FinalMag;
                mhi_dir(h,w) = FinalDir;
                mhi_X(h,w) = runningX*255/1024;
                mhi_Y(h,w) = runningY*255/1024;
            end
        end
       
        if (0 && theta==4)
            [Xs,Ys] = meshgrid(1:26,1:30);
            quiver(Xs,flipud(Ys),mhi_X,mhi_Y);  
            axis([1 26 1 30]); axis square
            str = sprintf('C:\\Data\\Users\\L623945\\face_database\\GEMEP\\emotion_test\\test\\test_044\\sample_images\\mhi\\\\mhi_quiver_theta%d_offset%d.png',theta,i);
            eval(['print -dpng -r36 ' str]);
            %print -djpeg C:\Data\Users\L623945\face_database\GEMEP\emotion_test\test\test_044\sample_images\mhi_ex_frames8_15_quiver.jpg
        end
        
        if mhi_options.thetaList(thetanum) == -1
            eval(['mhi_features.mhi_mag_frame_Theta9999(i,:,:) = mhi_mag(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
            eval(['mhi_features.mhi_dir_frame_Theta9999(i,:,:) = mhi_dir(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
            eval(['mhi_features.mhi_X_frame_Theta9999(i,:,:) = mhi_X(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
            eval(['mhi_features.mhi_Y_frame_Theta9999(i,:,:) = mhi_Y(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
        else
            eval(['mhi_features.mhi_mag_frame_Theta' num2str(theta) '(i,:,:) = mhi_mag(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
            eval(['mhi_features.mhi_dir_frame_Theta' num2str(theta) '(i,:,:) = mhi_dir(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
            eval(['mhi_features.mhi_X_frame_Theta' num2str(theta) '(i,:,:) = mhi_X(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
            eval(['mhi_features.mhi_Y_frame_Theta' num2str(theta) '(i,:,:) = mhi_Y(4:(WorkingHeight-3),4:(WorkingWidth-3));']);
        end
    end
end
%%
if (0)
    testimgin = zeros(FrameHeight,FrameWidth);
    testimgin(:) = mhiframe_Theta20(4,:,:);imshow(uint8(testimgin));
%     testimgin(10:30,5:15)=255;
%     testimgin(30:40,5:25)=255;
%     testimgin(10:35,25:40)=0;
    figure; imshow(uint8(testimgin));
    if (0) 
        print -djpeg C:\Data\Users\L623945\research\published_work\2012_ICIP_Paper\mhi_subject7_theta20_window4.jpg
    end
            
    filt = fspecial('gaussian',mhi_options.gaussianWidth*2,mhi_options.gaussianSigma*2);
    testimgin_save = testimgin;
    testimgin = imfilter(testimgin,filt);
    testimgin = imresize(testimgin,0.5);
    
    testimg_mag = zeros(round(FrameHeight/2)-6,round(FrameWidth/2)-6);
    testimg_mag(:) = mhi_features.mhi_mag_frame_Theta20(4,:,:);
    testimg_dir = zeros(round(FrameHeight/2)-6,round(FrameWidth/2)-6);
    testimg_dir(:) = mhi_features.mhi_dir_frame_Theta20(4,:,:);
    figure
    imshow(uint8(testimgin));
    
    magimg = uint8(testimg_mag*255/360);
    dirimg = uint8(testimg_dir*255/360);
    figure; imshow(magimg);
    if (0) 
        print -djpeg C:\Data\Users\L623945\research\published_work\2012_ICIP_Paper\mhi_magimg_subject7_theta20_window4.jpg
    end
    figure; imshow(dirimg);
    if (0) 
        print -djpeg C:\Data\Users\L623945\research\published_work\2012_ICIP_Paper\mhi_dirimg_subject7_theta20_window4.jpg
    end
    
    [numframes,WorkingHeight,WorkingWidth] = size(mhi_features.mhi_X_frame_Theta20);
    [Xq,Yq] = meshgrid(-WorkingWidth/2:WorkingWidth/2-1,-WorkingHeight/2:WorkingHeight/2-1);
    testimg_X = zeros(WorkingHeight,WorkingWidth); testimg_Y = zeros(WorkingHeight,WorkingWidth);
    testimg_X(:) = mhi_features.mhi_X_frame_Theta16(3,:,:);
    testimg_Y(:) = mhi_features.mhi_Y_frame_Theta16(3,:,:);
    figure; quiver(Xq,flipud(Yq),testimg_X,testimg_Y);
    if (0) 
        print -djpeg C:\Data\Users\L623945\research\published_work\2012_ICIP_Paper\mhi_XYmotion_subject7_theta20_window4.jpg
    end
    
    figure
    magheight = sqrt(testimg_X.^2 + testimg_Y.^2);
    contour(Xq,Yq,magheight,5);
    [dx,dy] = gradient(magheight,0.5,0.5);
    hold on
    quiver(Xq,flipud(Yq),dx,dy);
    
    figure
    magheight = sqrt(testimg_X.^2 + testimg_Y.^2);
    surf(Xq,Yq,magheight);
    [dx,dy] = gradient(magheight,0.5,0.5);
    hold on
    quiver(Xq,flipud(Yq),dx,dy);
    
    figure
    magheight = sqrt(testimg_X.^2 + testimg_Y.^2);
    contourf(Xq,Yq,magheight,5);
    [dx,dy] = gradient(magheight,0.5,0.5);
    hold on
    quiver(Xq,flipud(Yq),dx,dy,'k','linewidth',2);
end

%% Turn MHI image into motion vectors- orig method
% for each pixel that is not the brightest or darkest in the image, search
% neighborhood for brightest pixel that is connected in a monotone or
% continuously increasing fashion as we go from center outward
if (0)
eval(['maxsubplot = size(mframe_Theta' num2str(mhi_options.thetaList(end)) ',1);']);
testface = zeros(FrameHeight,FrameWidth);
mhi_dir = zeros(FrameHeight,FrameWidth);
mhi_mag = zeros(FrameHeight,FrameWidth);
dirLUT1 = [3*pi/4 pi/2  pi/4 pi 0 5*pi/4 3*pi/2 7*pi/4];
dirLUT2 = [ 3*pi/4  5*pi/8  pi/2   3*pi/8    pi/4; ...
            7*pi/8  3*pi/4  pi/2     pi/4    pi/8; ...
              pi      pi     0        0       0; ...
            9*pi/8  5*pi/4 3*pi/2  7*pi/4 15*pi/8; ...
            5*pi/4 11*pi/8 3*pi/2 13*pi/8  7*pi/4];
neighborlist5 = zeros(5,5);
neighborlist3 = zeros(3,3);
mask3 = zeros(3,3);
mask5=zeros(5,5);
i=1; thetanum=1;
for i=1:maxsubplot
    for thetanum=1:length(mhi_options.thetaList)
        theta = mhi_options.thetaList(1);
        eval(['testface(:) = mhiframe_Theta' num2str(theta) '(i,:,:);']);
        h=3; w=3;
        for h=3:(FrameHeight-2)  
            for w=3:(FrameWidth-2)
    %             neighborlist = [testface(h-1,w-1) testface(h-1,w) testface(h-1,w+ 1) ...
    %                 testface(h,w-1)  testface(h,w+ 1) ...
    %                 testface(h+1,w-1) testface(h+1,w) testface(h+1,w+ 1)];
    %             mval = max(neighborlist);
                neighborlist5 = testface(h-2:h+2,w-2:w+2);
                neighborlist3 = testface(h-1:h+1,w-1:w+1);
                mval3 = max(max(neighborlist3));
                [r3i,c3i] = find(neighborlist3 == mval3);
                current = testface(h,w);
                if (mval3 <= current)
                    %we must be at a bright maximum...so...no mag or dir info
                    mhi_mag(h,w) = 0;
                    mhi_dir(h,w) = 0;
                else
                    %magnitude different between center and max pixel
                    mhi_mag(h,w) = mval3(1)-current;
                    %if only one max in neighborhood, direction simple
                    if length(mval) == 1
                        mhi_dir(h,w) = dirLUT2(r3i+1,c3i+1);
                    else  
                        %we have multiple maxes and we need to see if one is
                        %better than the other...
                        %in 3x3 mask, mark all max values as 1
                        for ii=1:length(r3i)
                            mask3(r3i(ii),c3i(ii)) = 1;
                        end
                        %then dialate those max values to 5x5 area
                        mask5(2:4,2:4) = mask3;
                        sq3 = strel('square',3);
                        mask5 = imdilate(mask5,sq3);
                        %only use valid (dialated) locations...
                        [r5i,c5i] = find(mask5 == 0);
                        for ii=1:length(r5i)
                            neighborlist5(r5i(ii),c5i(ii)) = 0;
                        end
                        %zero out center 3x3 so it cannot participate
                        neighborlist5(2:4,2:4)=0;

                        %now max of remaining cells is a predictor of
                        %direction
                        mval5 = max(max(neighborlist5));
                        [r5i2,c5i2] = find(neighborlist5 == mval5);
                        if length(r5i2) == 1
                            mhi_dir(h,w) = dirLUT2(r5i2,c5i2);
                        else
                            %another tie...just take average...
                            mhi_dir(h,w) = dirLUT2(round(mean(r5i2)),round(mean(c5i2)));
                        end    

                    end
                end

            end
        end
        eval(['mhi_mag_frame_Theta' num2str(theta) '(i,:,:) = mhi_mag;']);
        eval(['mhi_dir_frame_Theta' num2str(theta) '(i,:,:) = mhi_dir;']);
    end
end
end

%% Normalize mframe_Theta* magnitude data
% Note mhiframe_Theta* is already normalized...
if (0)
eval(['maxsubplot = size(mframe_Theta' num2str(mhi_options.thetaList(end)) ',1);']);
testface = zeros(FrameHeight,FrameWidth);
theta = mhi_options.thetaList(end);
runningmax=0;
i=1;
for i=1:maxsubplot
    eval(['testface(:) = mframe_Theta' num2str(theta) '(i,:,:);']);
    maxpix = max(max(testface));
    if maxpix > runningmax
        runningmax = maxpix
    end
end

for thetanum=1:length(mhi_options.thetaList)
    theta = mhi_options.thetaList(thetanum);
    eval(['mframe_Theta' num2str(theta) '= mframe_Theta' num2str(theta) '.*255./' num2str(runningmax) ';']);
end
end


%%  Shows all the difference images- OLD code
if (0) %orig plots when mframe_Theta* was different lengths
for thetanum=1:length(mhi_options.thetaList)
    figure
    testface = zeros(FrameHeight,FrameWidth);
    theta = mhi_options.thetaList(thetanum);
    eval(['maxsubplot = size(mframe_Theta' num2str(theta) ',1)']);
    for i=1:maxsubplot
        subplot(5,5,i);
        eval(['testface(:) = mframe_Theta' num2str(theta) '(i,:,:);']);
        imagesc(testface);
    end
end
end

%% Show only the center difference images- OLD code
if (0) %orig plots when mframe_Theta* was different lengths, all on 1 plot, centered on common t
eval(['maxsubplot = size(mframe_Theta' num2str(mhi_options.thetaList(end)) ',1);']);
figure
for thetanum=1:length(mhi_options.thetaList)   
    testface = zeros(FrameHeight,FrameWidth);
    theta = mhi_options.thetaList(thetanum);
    thetamax = mhi_options.thetaList(end);
    startframe = (thetamax-theta)/2;
    for i=1:maxsubplot
        subplot(length(mhi_options.thetaList),maxsubplot,i+(thetanum-1)*maxsubplot);
        eval(['testface(:) = mframe_Theta' num2str(theta) '(i+startframe,:,:);']);
        imagesc(testface);
    end
end
end
%% Show only the center difference images- Good code
if (0)
eval(['maxsubplot = size(mframe_Theta' num2str(mhi_options.thetaList(end)) ',1);']);
testface = zeros(FrameHeight,FrameWidth);
figure
for thetanum=1:length(mhi_options.thetaList)   
    theta = mhi_options.thetaList(thetanum);
    for i=1:maxsubplot
        subplot(length(mhi_options.thetaList),maxsubplot,i+(thetanum-1)*maxsubplot);
        eval(['testface(:) = mhiframe_Theta' num2str(theta) '(i,:,:);']);
        %imagesc(testface);
        imshow(uint8(testface))
    end
end
figure
for thetanum=1:length(mhi_options.thetaList)   
    theta = mhi_options.thetaList(thetanum);
    for i=1:maxsubplot
        subplot(length(mhi_options.thetaList),maxsubplot,i+(thetanum-1)*maxsubplot);
        eval(['testface(:) = mframe_Theta' num2str(theta) '(i,:,:);']);
        %imagesc(testface);
        imshow(uint8(testface))
    end
end
end
% testface(:) = dframeOpen(3,:,:);

% testface = zeros(FrameHeight,FrameWidth);
% testface(:) = dframeOpen(3,:,:);
% figure; imagesc(uint8(testface)); 
