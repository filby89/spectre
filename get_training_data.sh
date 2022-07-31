#!/bin/bash
# file adapted from MICA https://raw.githubusercontent.com/Zielon/MICA
#
echo -e "\nDownloading deca_model..."
#
FILEID=1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje
FILENAME=./data/deca_model.tar
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt


echo "To download the Emotion Recognition from EMOCA which is used from SPECTRE for expression loss, please register at:",
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/ResNet50.zip -O ResNet50.zip
unzip ResNet50.zip -d data/
rm ResNet50.zip

echo -e "\nDownloading lipreading pretrained model..."

FILEID=1yHd4QwC7K_9Ro2OM_hC7pKUT2URPvm_f
FILENAME=LRS3_V_WER32.3.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
unzip $FILENAME -d data/
rm LRS3_V_WER32.3.zip

echo -e "\nDownloading landmarks for LRS3 dataset ..."

gdown --id 1QRdOgeHvmKK8t4hsceFVf_BSpidQfUyW
unzip LRS3_landmarks.zip -d data/
rm LRS3_landmarks.zip



echo -e "\nInstallation has finished!"
