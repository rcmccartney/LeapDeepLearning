function image = ImageMaker_new(filename, dim, sampleSize, linesToSkip,...
                    numfingers, numdims)

% dim is size of produced images
%open the current file
d = fopen(filename);
%error check
if d == -1
    error('Data file cannot be open - ERROR!')
end

%move position to beginning of matrix, skipping the header lines
for i=1:linesToSkip,
    fgetl(d);
end;

total = numfingers * numdims;
p = zeros(dim,dim,total);
sizep = size(squeeze(p(:,:,1)));
% the 15 images after being resized will be stored here
q = zeros(sampleSize,sampleSize,total); 
% loop through file grabbing the finger positions
line = fgetl(d);
while line ~= -1,
    out = regexp(line, ' *', 'split');
    for i=1:numfingers,
        % increment by numdims for each of the planes to be included
        % since we are storing the images in a flat array
        index = numdims*(i-1);
        % get this line of data for this finger
        mat = fingers(out, dim, i);
        mat = mat(~any(isnan(mat),2),:);
        % calculate the linear indices
        xy_indices = sub2ind(sizep, mat(:,1), mat(:,2));    
        xz_indices = sub2ind(sizep, mat(:,1), mat(:,3));
        yz_indices = sub2ind(sizep, mat(:,2), mat(:,3));
        indices = [xy_indices xz_indices yz_indices];
        for j=1:numdims,
            curr = index + j;
            % grab each finger picture out of full array - no decay used here
            p_j = squeeze(p(:,:,curr));
            % indexing into the image and update pixels
            % before used p_j(indices(:,j)) + 1 to get differing intensities
            % here only using 0,1 as binary pixel activation
            p_j(indices(:,j)) = 1;
            % stick back into full array 
            p(:,:,curr) = p_j;
        end;
    end;
    % go to next line
    line = fgetl(d);
end;

% look at the gesture before resizing
% preimage = reshape(p, [dim, dim*numdims]);
% figure(1), imshow(imrotate(preimage,90));

for i=1:total,
    % crop to only the gesture itself
    temp = squeeze(p(:,:,i));
    [row, col] = find(temp);   
    temp = temp(min(row):max(row),min(col):max(col));
    if isempty(temp),
        q(:,:,i) = zeros(sampleSize, sampleSize);  
    else
        q(:,:,i) = imresize(temp, [sampleSize sampleSize]);
    end;
end;
      
image = reshape(q, [sampleSize, sampleSize*numdims]);
% need to make image in range 0 to 1
% image = mat2gray(image);

% normalize pixel activations
% dataMean = mean(image(:));
% image = bsxfun(@minus, image, dataMean) ;

% this visualizes the produced image
% figure(2), imshow(imrotate(image,90));

%close file
fclose(d);