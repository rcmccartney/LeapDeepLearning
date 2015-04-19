% runs the declared CNN network and then visualizes the 
% first layer of features

addpath code
network = 'cnn_leap_binary' ;

run(network) ;
image = [] ;

for i = 1:5,
  image = [ image; ...
            x.layers{1}.filters(:, :, 1, 4*(i-1) + 1 ) ... 
            x.layers{1}.filters(:, :, 1, 4*(i-1) + 2 ) ... 
            x.layers{1}.filters(:, :, 1, 4*(i-1) + 3 ) ...
            x.layers{1}.filters(:, :, 1, 4*(i-1) + 4 ) ]
end;

imshow(image)