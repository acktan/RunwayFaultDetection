

def make_prediction(IDs, path, model, thresh, path_to_saved_submissions_folder, submission_name, save=True):
    for idx, ID in enumerate(IDs):
        img_path = os.path.join(path, str(ID))
        trues_values = labels_train[labels_train['filename'] == str(ID)].drop('filename', axis=1).values

        # Read and prepare image
        img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
        img = image.img_to_array(img)
        img = img/255
        img = np.expand_dims(img, axis=0)

        # Generate prediction
        prediction = (model.predict(img) > thresh).astype('int')[0]
        if idx == 0:
            predictions = prediction
        else:
            predictions = np.vstack([predictions, prediction])

    


    predictions = make_prediction(template_test.filename.values,
                                  repo+'hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/test',
                                  model, 
                                  thresh=PREDICTION_THRESH,
                                 )

    df = pd.DataFrame(predictions, columns=template_test.columns[1:])
    df['filename'] = template_test.filename
    df = df[['filename', 
             'FISSURE', 
             'REPARATION', 
             'FISSURE LONGITUDINALE', 
             'MISE EN DALLE']]
    
    if save=True:
        df.to_csv(path_to_saved_submissions_folder repo  + submission_name, index=False)
    
    return df