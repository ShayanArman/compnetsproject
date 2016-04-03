%converts images to array for input into neurons
function [test_data] = images2netinput(test_images)
    inputSize = numel(cell2mat(test_images(1)));
    test_data = zeros(inputSize,numel(test_images));
    for i = 1:numel(test_images)
        test_data(:,i) = test_images{i}(:);
    end
end
