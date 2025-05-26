import json

bus_classes = ['benign', 'malignant', 'normal']

aml_classes = ['BAS Basophil', 'EBO Erythroblast', 'EOS Eosinophil',
       'KSC Smudge cell', 'LYA Lymphocyte (atypical)',
       'LYT Lymphocyte (typical)', 'MMZ Metamyelocyte', 'MOB Monoblast',
       'MON Monocyte', 'MYB Myelocyte', 'MYO Myeloblast',
       'NGB Neutrophil (band)', 'NGS Neutrophil (segmented)',
       'PMB Promyelocyte (bilobled)', 'PMO Promyelocyte']

crc_classes = ['adipose (ADI)', 'background (BACK)', 'debris (DEB)',
       'lymphocytes (LYM)', 'mucus (MUC)', 'smooth muscle (MUS)',
       'normal colon mucosa (NORM)', 'cancer-associated stroma (STR)',
       'colorectal adenocarcinoma epithelium (TUM)']

cxr_classes = ['Hernia', 'Emphysema', 'Mass', 'Cardiomegaly', 'Pneumothorax', 
            'Edema', 'Effusion', 'Pneumonia', 'Nodule', 'Pleural_Thickening', 
            'Fibrosis', 'Consolidation', 'Infiltration', 'Atelectasis', 'None']

derm_classes = ['Melanocytic nevus', 'Melanoma',
       'Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)',
       'Dermatofibroma',
       'Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)',
       'Basal cell carcinoma', 'Vascular lesion']

dr_regular_classes = ['Quality is not good enough for the diagnosis of retinal diseases',
                        'Quality is good enough for the diagnosis of retinal diseases']

fundus_classes = ['abnormal', 'normal']
    
glaucoma_classes = ['Normal', 'Suspect']

mammo_calc_classes = ['malignant', 'benign']

mammo_mass_classes = ['malignant', 'benign']


oct_classes = ['CNV', 'NORMAL', 'DME', 'DRUSEN']

organs_axial_classes = ['liver', 'right kidney', 'left kidney', 'right femoral head',
                    'left femoral head', 'bladder', 'spleen', 'pancreas', 'heart',
                    'right lung', 'left lung']

organs_coronal_classes = ['liver', 'right kidney', 'left kidney', 'right femoral head',
                    'left femoral head', 'bladder', 'spleen', 'pancreas', 'heart',
                    'right lung', 'left lung']

organs_sagittal_classes = ['liver', 'right kidney', 'left kidney', 'right femoral head',
                        'left femoral head', 'bladder', 'spleen', 'pancreas', 'heart',
                        'right lung', 'left lung']

pbc_classes = ['basophil', 'eosinophil', 'erythroblast', 'immature granulocyte',
            'lymphocyte', 'monocyte', 'neutrophil', 'platelet']      

pneumonia_classes = ['NORMAL', 'BACTERIA', 'VIRUS']    

skinl_derm_classes = ['basal cell carcinoma', 'blue nevus', 'clark nevus',
       'combined nevus', 'congenital nevus', 'dermal nevus',
       'dermatofibroma', 'lentigo', 'melanoma (in situ)',
       'melanoma (less than 0.76 mm)', 'melanoma (0.76 to 1.5 mm)',
       'melanoma (more than 1.5 mm)', 'melanoma metastasis', 'melanosis',
       'miscellaneous', 'recurrent nevus', 'reed or spitz nevus',
       'seborrheic keratosis', 'vascular lesion', 'melanoma']

skinl_photo_classes = ['basal cell carcinoma', 'blue nevus', 'clark nevus',
       'combined nevus', 'congenital nevus', 'dermal nevus',
       'dermatofibroma', 'lentigo', 'melanoma (in situ)',
       'melanoma (less than 0.76 mm)', 'melanoma (0.76 to 1.5 mm)',
       'melanoma (more than 1.5 mm)', 'melanoma metastasis', 'melanosis',
       'miscellaneous', 'recurrent nevus', 'reed or spitz nevus',
       'seborrheic keratosis', 'vascular lesion', 'melanoma']


# MedMNIST 12x datasets
tissuemnist_classes = ['Collecting Duct, Connecting Tubule', 'Distal Convoluted Tubule', 'Glomerular endothelial cells', 'Interstitial endothelial cells', 'Leukocytes', 'Podocytes', 'Proximal Tubule Segments', 'Thick Ascending Limb']
pathmnist_classes = ['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle', 'normal colon mucosa', 'cancer-associated stroma', 'colorectal adenocarcinoma epithelium']
chestmnist_classes = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation', 'edema', 'emphysema', 'fibrosis', 'pleural', 'hernia']
dermamnist_classes = ['actinic keratoses and intraepithelial carcinoma', 'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions']
octmnist_classes = ['choroidal neovascularization', 'diabetic macular edema', 'drusen', 'normal']
pneumoniamnist_classes = ['normal', 'pneumonia']
retinamnist_classes = ['0', '1', '2', '3', '4']
breastmnist_classes = ['malignant', 'normal, benign']
bloodmnist_classes = ['basophil', 'eosinophil', 'erythroblast', 'immature granulocytes(myelocytes, metamyelocytes and promyelocytes)', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
organamnist_classes = ['bladder', 'femur-left', 'femur-right', 'heart', 'kidney-left', 'kidney-right', 'liver', 'lung-left', 'lung-right', 'pancreas', 'spleen']
organcmnist_classes = ['bladder', 'femur-left', 'femur-right', 'heart', 'kidney-left', 'kidney-right', 'liver', 'lung-left', 'lung-right', 'pancreas', 'spleen']
organsmnist_classes = ['bladder', 'femur-left', 'femur-right', 'heart', 'kidney-left', 'kidney-right', 'liver', 'lung-left', 'lung-right', 'pancreas', 'spleen']

modalities = {
    'pathmnist': 'colon pathology', 
    'octmnist': 'retinal OCT', 
    'pneumoniamnist': 'chest x-ray', 
    'retinamnist': 'fundus camera', 
    'breastmnist': 'breast ultrasound' 
}