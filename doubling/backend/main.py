from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd, numpy as np, io
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app=FastAPI(title="DoublingIQ API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# DEMO / SYNTHETIC DATASET
rng=np.random.default_rng(42)
machines=[f"M{i:02d}" for i in range(1,6)]
rows=[]
for i in range(300):
    m=rng.choice(machines); shift=rng.choice(["Morning","Evening","Night"])
    h=rng.uniform(5,12); prod=rng.uniform(180,500); prev=rng.uniform(55,125)
    down=rng.uniform(0,2); maint=int(rng.integers(0,8)); age=int(rng.integers(1,16))
    energy=4.8*h+.105*prod+.16*prev+2.5*down+.7*maint+.45*age+machines.index(m)*3+rng.normal(0,4)
    rows.append(dict(machine_id=m,date=f"2026-08-{i%28+1:02d}",shift=shift,
        operating_hours=round(h,2),production_quantity=round(prod,2),
        energy_consumption_kwh=round(max(20,energy),2),downtime_hours=round(down,2),
        maintenance_count=maint,machine_age=age))
df=pd.DataFrame(rows)
df["previous_energy"]=df.energy_consumption_kwh.shift(1).bfill()
features=["machine_id","shift","operating_hours","production_quantity","previous_energy","downtime_hours","maintenance_count","machine_age"]
cat=["machine_id","shift"]; num=[x for x in features if x not in cat]
pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
model=Pipeline([("pre",pre),("rf",RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1))])
a,b=train_test_split(df,test_size=.2,random_state=42)
model.fit(a[features],a.energy_consumption_kwh)
p=model.predict(b[features])
metrics={"MAE":round(float(mean_absolute_error(b.energy_consumption_kwh,p)),3),
"RMSE":round(float(np.sqrt(mean_squared_error(b.energy_consumption_kwh,p))),3),
"R2":round(float(r2_score(b.energy_consumption_kwh,p)),3)}

class Prediction(BaseModel):
    machine_id:str; date:str; shift:str
    operating_hours:float=Field(gt=0); expected_production:float=Field(gt=0)
    previous_energy:float=Field(ge=0); downtime:float=Field(ge=0)
    maintenance_count:int=Field(ge=0); machine_age:int=Field(ge=0)

@app.get("/api/health")
def health(): return {"status":"ok","dataset":"DEMO / SYNTHETIC DATASET"}

@app.get("/api/metrics")
def get_metrics(): return metrics

@app.get("/api/dashboard")
def dashboard():
    d=df.tail(20)
    return {"dataset":"DEMO / SYNTHETIC DATASET","total_machines":24,
    "today_energy":round(float(d.energy_consumption_kwh.sum()),1),
    "predicted_energy":round(float(d.energy_consumption_kwh.sum()*.96),1),
    "average_efficiency":round(float((d.energy_consumption_kwh/d.production_quantity).mean()),3),
    "high_consumption_machines":3,"today_production":round(float(d.production_quantity.sum()),1)}

@app.get("/api/machines")
def get_machines():
    out=[]
    for m in machines:
        d=df[df.machine_id==m]
        out.append({"machine_id":m,"energy":round(float(d.energy_consumption_kwh.mean()),2),
        "production":round(float(d.production_quantity.mean()),2),
        "efficiency":round(float((d.energy_consumption_kwh/d.production_quantity).mean()),3)})
    return out

@app.get("/api/history")
def history():
    d=df.tail(50).copy()
    d["efficiency"]=d.energy_consumption_kwh/d.production_quantity
    return d.round(3).to_dict("records")

@app.post("/api/predict")
def predict(x:Prediction):
    q=pd.DataFrame([{"machine_id":x.machine_id,"shift":x.shift,"operating_hours":x.operating_hours,
    "production_quantity":x.expected_production,"previous_energy":x.previous_energy,
    "downtime_hours":x.downtime,"maintenance_count":x.maintenance_count,"machine_age":x.machine_age}])
    y=float(model.predict(q)[0]); eff=y/x.expected_production
    return {"machine":x.machine_id,"predicted_kwh":round(y,2),
    "range":[round(y*.9,2),round(y*1.1,2)],"efficiency":round(eff,3),
    "status":"NORMAL","dataset":"DEMO / SYNTHETIC DATASET"}

@app.post("/api/upload")
async def upload(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file")
    try: d=pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e: raise HTTPException(400,f"Invalid CSV: {e}")
    required={"machine_id","date","shift","operating_hours","production_quantity",
              "energy_consumption_kwh","downtime_hours","maintenance_count","machine_age"}
    missing=sorted(required-set(d.columns))
    if missing: raise HTTPException(400,"Missing columns: "+", ".join(missing))
    return {"rows":len(d),"message":"CSV validated successfully","columns":list(d.columns)}
