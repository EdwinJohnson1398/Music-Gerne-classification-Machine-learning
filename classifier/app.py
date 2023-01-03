
import numpy as np
from flask import Flask, request, render_template
import pickle
import librosa
import pandas as pd
import catboost as cb
import xgboost as xgb
from os import path
from pydub import AudioSegment




#UPLOAD_FILE = '/Users/Anil/mpr/'
#UPLOAD_FOLDER = '/Data/'
ALLOWED_EXTENSIONS = set(['wav','mp3'])

app = Flask(__name__)

#app.config['UPLOAD_FILE'] = UPLOAD_FILE

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')








@app.route('/predict',methods=['POST'])
#@app.route('/find' , methods = ['GET' , 'POST'])
def find():
    if request.method == 'GET':
        return "HELLO"

@app.route('/' , methods = ['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        print("file loaded")
        

        
        if file and allowed_file(file.filename):

            audio_data, sr =librosa.load(file) #, offset=0, duration=30)
            audio_data, _ = librosa.effects.trim(audio_data)
            audio_data = audio_data[:661500]
            collection = np.split(audio_data,10)
            

            audio_data = collection[0]
            
            d = librosa.feature.mfcc(np.array(audio_data).flatten(),sr=22050 , n_mfcc = 20) #36565
            d_var = d.var(axis=1).tolist()
            d_mean = d.mean(axis=1).tolist()
            test_data = []#[d_var + d_mean]
            for i in range(20):
                test_data.append(d_mean[i])
                test_data.append(d_var[i])
                mfcc_names=[]
            for i in range(1,21):
                mfcc_str = "mfcc"+str(i)+"_mean"
                mfcc_names.append(mfcc_str)
                mfcc_str = "mfcc"+str(i)+"_var"
                mfcc_names.append(mfcc_str)
            test_frame = pd.DataFrame([test_data], columns = mfcc_names)
            test_data = []
            mfcc_names=[]
            #chroma
            S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
            chroma = librosa.feature.chroma_stft(S=S, sr=sr)
            #chroma_stft_mean
            chroma_mean = round(np.mean(chroma),6)
            test_data.append(chroma_mean)
            #chrome_stft_var
            chroma_var = round(np.var(chroma),6)
            test_data.append(chroma_var)
            #chroma_label
            mfcc_names.append("chroma_stft_mean")
            mfcc_names.append("chroma_stft_var")

            #rms
            rms = librosa.feature.rms(y=audio_data)
            #rms_mean
            rms_mean = round(np.mean(rms),6)
            test_data.append(rms_mean)
            #rms_var
            rms_var = round(np.var(rms),6)
            test_data.append(rms_var)
            #rms_label
            mfcc_names.append("rms_mean")
            mfcc_names.append("rms_var")

            #spectral_centroid
            cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            #spectral_centroid_mean
            sc_mean = round(np.mean(cent),6)
            test_data.append(sc_mean)
            #spectral_centroid_var
            sc_var = round(np.var(cent),6)
            test_data.append(sc_var)
            #sc_label
            mfcc_names.append("spectral_centroid_mean")
            mfcc_names.append("spectral_centroid_var")

            #spectral_bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            #spectral_bandwidth_mean
            spec_bw_mean = round(np.mean(spec_bw),6)
            test_data.append(spec_bw_mean)
            #spectral_bandwidth_var
            spec_bw_var = round(np.var(spec_bw),6)
            test_data.append(spec_bw_var)
            #sb_label
            mfcc_names.append("spectral_bandwidth_mean")
            mfcc_names.append("spectral_bandwidth_var")

            #rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            #rolloff_mean
            rolloff_mean = round(np.mean(rolloff),6)
            test_data.append(rolloff_mean)
            #rolloff_var
            rolloff_var = round(np.var(rolloff),6)
            test_data.append(rolloff_var)
            #rolloff_label
            mfcc_names.append("rolloff_mean")
            mfcc_names.append("rolloff_var")

            #zero_crossing_rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            #zero_crossing_rate_mean
            zcr_mean = round(np.mean(zcr),6)
            test_data.append(zcr_mean)
            #zero_crossing_rate_var
            zcr_var = round(np.var(zcr),6)
            test_data.append(zcr_var)
            #zero_crossing_rate_label
            mfcc_names.append("zero_crossing_rate_mean")
            mfcc_names.append("zero_crossing_rate_var")

           


            #tempo
            hop_length = 512
            oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
            tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                    hop_length=hop_length)[0]

            tempo = round(tempo,6)
            test_data.append(tempo)
            #tempo_label
            mfcc_names.append("tempo")
            d_var = d.var(axis=1).tolist()
            d_mean = d.mean(axis=1).tolist()
            #test_data = []#[d_var + d_mean]
            for i in range(20):
                test_data.append(d_mean[i])
                test_data.append(d_var[i])
            for i in range(1,21):
                mfcc_str = "mfcc"+str(i)+"_mean"
                mfcc_names.append(mfcc_str)
                mfcc_str = "mfcc"+str(i)+"_var"
                mfcc_names.append(mfcc_str)

            
            scaler = pickle.load(open('scalar.pkl', 'rb'))
            X_train = pickle.load(open('xtrain.pkl', 'rb'))
           
            test_frame = pd.DataFrame([test_data], columns = mfcc_names)
            testing_frame = pd.DataFrame(scaler.transform(test_frame), columns=X_train.columns)
            shorter_testing_frame = testing_frame

            print("------------------middle1-------------------------")
            
            val=1
            while(val<=9):

                audio_data = collection[val]
                d = librosa.feature.mfcc(np.array(audio_data).flatten(),sr=22050 , n_mfcc = 20) #36565
                d_var = d.var(axis=1).tolist()
                d_mean = d.mean(axis=1).tolist()
                test_data = []#[d_var + d_mean]
                for i in range(20):
                    test_data.append(d_mean[i])
                    test_data.append(d_var[i])
                mfcc_names=[]
                for i in range(1,21):
                    mfcc_str = "mfcc"+str(i)+"_mean"
                    mfcc_names.append(mfcc_str)
                    mfcc_str = "mfcc"+str(i)+"_var"
                    mfcc_names.append(mfcc_str)
                test_frame = pd.DataFrame([test_data], columns = mfcc_names)
                test_data = []
                mfcc_names=[]
                #chroma
                S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
                chroma = librosa.feature.chroma_stft(S=S, sr=sr)
                #chroma_stft_mean
                chroma_mean = round(np.mean(chroma),6)
                test_data.append(chroma_mean)
                #chrome_stft_var
                chroma_var = round(np.var(chroma),6)
                test_data.append(chroma_var)
                #chroma_label
                mfcc_names.append("chroma_stft_mean")
                mfcc_names.append("chroma_stft_var")

                #rms
                rms = librosa.feature.rms(y=audio_data)
                #rms_mean
                rms_mean = round(np.mean(rms),6)
                test_data.append(rms_mean)
                #rms_var
                rms_var = round(np.var(rms),6)
                test_data.append(rms_var)
                #rms_label
                mfcc_names.append("rms_mean")
                mfcc_names.append("rms_var")

                #spectral_centroid
                cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
                #spectral_centroid_mean
                sc_mean = round(np.mean(cent),6)
                test_data.append(sc_mean)
                #spectral_centroid_var
                sc_var = round(np.var(cent),6)
                test_data.append(sc_var)
                #sc_label
                mfcc_names.append("spectral_centroid_mean")
                mfcc_names.append("spectral_centroid_var")

                #spectral_bandwidth
                spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
                #spectral_bandwidth_mean
                spec_bw_mean = round(np.mean(spec_bw),6)
                test_data.append(spec_bw_mean)
                #spectral_bandwidth_var
                spec_bw_var = round(np.var(spec_bw),6)
                test_data.append(spec_bw_var)
                #sb_label
                mfcc_names.append("spectral_bandwidth_mean")
                mfcc_names.append("spectral_bandwidth_var")

                #rolloff
                rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
                #rolloff_mean
                rolloff_mean = round(np.mean(rolloff),6)
                test_data.append(rolloff_mean)
                #rolloff_var
                rolloff_var = round(np.var(rolloff),6)
                test_data.append(rolloff_var)
                #rolloff_label
                mfcc_names.append("rolloff_mean")
                mfcc_names.append("rolloff_var")

                #zero_crossing_rate
                zcr = librosa.feature.zero_crossing_rate(audio_data)
                #zero_crossing_rate_mean
                zcr_mean = round(np.mean(zcr),6)
                test_data.append(zcr_mean)
                #zero_crossing_rate_var
                zcr_var = round(np.var(zcr),6)
                test_data.append(zcr_var)
                #zero_crossing_rate_label
                mfcc_names.append("zero_crossing_rate_mean")
                mfcc_names.append("zero_crossing_rate_var")



                tempo
                hop_length = 512
                oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
                tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                            hop_length=hop_length)[0]

                tempo = round(tempo,6)
                test_data.append(tempo)
                #tempo_label
                mfcc_names.append("tempo")
                d_var = d.var(axis=1).tolist()
                d_mean = d.mean(axis=1).tolist()
                #test_data = []#[d_var + d_mean]
                for i in range(20):
                    test_data.append(d_mean[i])
                    test_data.append(d_var[i])
                for i in range(1,21):
                    mfcc_str = "mfcc"+str(i)+"_mean"
                    mfcc_names.append(mfcc_str)
                    mfcc_str = "mfcc"+str(i)+"_var"
                    mfcc_names.append(mfcc_str)


                test_frame2 = pd.DataFrame([test_data], columns = mfcc_names)
                testing_frame2 = pd.DataFrame(scaler.transform(test_frame2), columns=X_train.columns)
                shorter_testing_frame2 =testing_frame2
                df_test = pd.concat([shorter_testing_frame, shorter_testing_frame2])
                shorter_testing_frame = df_test
                val+=1
            # print(test_frame2.shape())
            # print(df_test)

            print(".................middle.............")
         
            gbc = pickle.load(open('gbc.pkl', 'rb'))
            abc = pickle.load(open('abc.pkl', 'rb'))
            rfc = pickle.load(open('rfc.pkl', 'rb'))
            lr = pickle.load(open('lr.pkl', 'rb'))
            cls = pickle.load(open('cls.pkl', 'rb'))

            #Testing Input Data
            #from collections import Counter
            result=[]
            result1=[]
            models = {'GradientBoostingClassifier':gbc ,'AdaBoost':abc,  'Linear Regression':lr, 'KNN':cls,'Random Forest':rfc }
            
            key_list = list(models.keys())
            val_list = list(models.values())


            for model in models.values():
                position = val_list.index(model)

                for i in range(10):
                    test = model.predict(df_test[i:(i+1)])
                    result.append(test)
                t = max(result, key = result.count)

                

                print(t)
                print("\n")
                print(result)
                print("\n")
                if t== [[0]] or t ==[['blues']]:
                    genre_detected = 'blues'
                elif t== [[1]] or t==[['Classical']]:
                    genre_detected = 'Classical'
                elif t== [[2]] or t==[['Country']]:
                    genre_detected = 'Country'
                elif t== [[3]] or t==[['Disco']]:
                    genre_detected = 'Disco'
                elif t== [[4]] or t==[['Hiphop']]:
                    genre_detected = 'Hiphop'
                elif t== [[5]] or t==[['jazz']]:
                    genre_detected = 'Jazz'
                elif t== [[6]] or t==[['Metal']]:
                    genre_detected = 'Metal'
                elif t== [[7]] or t==[['pop']]:
                    genre_detected = 'Pop'
                elif t== [[8]] or t==[['reggae']]:
                    genre_detected = 'Regage'
                else:
                    genre_detected = 'Rock'

                result1.append(genre_detected)
            print(result1)

            t1 = max(result1, key = result.count)
            n=result1.count(t1)
                
            if (n/5)*100 <80:
                return render_template('index.html' , prediction_text = "Unknown Audio")

            result2=result1[3]
            # result = result[:7]
            print(result)
            print("\n")
            #return genre_detected
            print(".........last.............")

            return render_template('index.html' , prediction_text = result2, algos = key_list )

        else:
            return render_template('index.html' , prediction_text = "Upload in wav or mp3 format")




if __name__ == "__main__":
  app.run(host='0.0.0.0',port=5000, debug=True)