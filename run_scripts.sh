cd fixed_source/square
python3 square.py | tee square.out
python3 postprocess.py
python3 plot_direction.py
python3 plot_meshsize.py
cd ../../

cd fixed_source/circle
python3 circle.py | tee circle.out
python3 postprocess.py
python3 plot_direction.py
python3 plot_meshsize.py
cd ../../

cd fixed_source/quarter_circle
python3 quarter_circle.py | tee quarter_circle.out
python3 postprocess.py
python3 plot_direction.py
python3 plot_meshsize.py
cd ../../

cd fixed_source/cruciform
python3 cruciform.py
python3 plot.py
cd ../../

cd eigenvalue/quarter_circle
python3 quarter_circle.py
python3 plot.py
cd ../../

cd eigenvalue/circle
python3 circle.py
cd ../../

cd eigenvalue/pincell
python3 pincell.py
python3 plot.py
cd ../../

cd eigenvalue/lightbridge_ba
python3 lightbridge_ba.py
python3 plot.py
cd ../../

cd eigenvalue/lightbridge_gas
python3 lightbridge_gas.py
cd ../../
