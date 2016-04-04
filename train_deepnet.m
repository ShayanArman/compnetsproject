%initialize and train the deepnet given data and number of epochs
function [deepnet] = train_deepnet(xTrainImages, tTrain, epochs)
    deepLayer1 = trainAutoencoder(xTrainImages,100, 'MaxEpochs', epochs, 'ShowProgressWindow',false);
    featLayer1 = encode(deepLayer1,xTrainImages);

    deepLayer2 = trainAutoencoder(featLayer1,50, 'MaxEpochs', epochs, 'ShowProgressWindow',false);
    featLayer2 = encode(deepLayer2,featLayer1);

    deepLayer3 = trainSoftmaxLayer(featLayer2,tTrain,'MaxEpochs', epochs,'ShowProgressWindow',false);
    
    deepnet = stack(deepLayer1,deepLayer2,deepLayer3);
    deepnet.trainParam.epochs = epochs;
    deepnet.trainParam.showWindow = false;
    deepnet = train(deepnet, images2netinput(xTrainImages), tTrain);
end

