import pickle
import sys



# ESR model 
with open('models/model_LR_esr.pkl','rb') as f:
    esr_model = pickle.load(f)
    print(esr_model.predict([[sys.argv[1]]]))

# Ja model 

with open('models/gradient_boosting_Ja.pkl','rb') as f:
    Ja_model = pickle.load(f)
    
    print(Ja_model.predict([[sys.argv[1]]]))

# Jn model 

with open('models/ababoost_Jn.pkl','rb') as f:
        Jn_model = pickle.load(f)
        pred = Jn_model.predict([[sys.argv[1]]])
        print(pred)

# Jr Model 
with open('models/Decision_tree_regressor_Jr.pkl','rb') as f:
        Jr_model = pickle.load(f)
        print(Jr_model.predict([[sys.argv[1]]]))
# Jw model 
with open('models/gradient_boosting_Jw.pkl','rb') as f:
        Jw_model = pickle.load(f)
        print(Jw_model.predict([[sys.argv[1]]]))

# MUS model 
with open('models/ababoost_us.pkl','rb') as f:
        mus_model = pickle.load(f)
        print(mus_model.predict([[sys.argv[1],	sys.argv[2]]]))

# Q model 
with open('models/q.pkl','rb') as f:
        q_model = pickle.load(f)
        print(q_model.predict([[sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]]))


# rmr model 
with open('models/rmr.pkl','rb') as f:
        rmr_model = pickle.load(f)
        print(rmr_model.predict([[sys.argv[1]]]))

# rqd model 
with open('models/rqd_GBR_rqd.pkl','rb') as f:
        RQD_model_GBR = pickle.load(f)
        print(RQD_model_GBR.predict([[sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]]))

# srf model 
with open('models/extra_tree_srf.pkl','rb') as f:
        srf_model = pickle.load(f)
        print(srf_model.predict([[sys.argv[1]]])) 

# ucs/vsr model 

with open('models/ucsvsr_update.pkl','rb') as f:
        ucsvsr_model = pickle.load(f)
        print(ucsvsr_model.predict([[sys.argv[1], sys.argv[2], sys.argv[3]]]))