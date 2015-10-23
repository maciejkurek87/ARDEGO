EASY_INSTALL=pip install
QUADS_DIR = examples/quadrature_method_based_app/fitness_script.py
SCRIPTS2D = configuration_matern configuration_ani configuration_iso
SCRIPTS2D001 = configuration_2d_matern configuration_2d_ani configuration_2d_iso
SCRIPTS3D = configuration_3d_matern configuration_3d_ani configuration_3d_iso
SCRIPTS3D001 = configuration_3d001_matern configuration_3d001_ani configuration_3d3d001_ani

rungui:
	./run_mlo.sh --gui

run_example1:
	./run_mlo.sh -f examples/artificial_continous_function/fitness_script.py -c examples/artificial_continous_function/configuration_script.py

run_debug:
	./run_debug.sh -f examples/quadrature_method_based_app/fitness_script.py -c examples/quadrature_method_based_app/configuration_script.py

run_example2:
	./run_mlo.sh -f examples/quadrature_method_based_app/fitness_script.py -c examples/quadrature_method_based_app/configuration_script.py

run_example3:
	./run_mlo.sh -f examples/reconfigurable_radio/fitness_script.py -c examples/reconfigurable_radio/configuration_script.py

run_example4:
	./run_mlo.sh -f examples/xinyu_rtm/fitness_script.py -c examples/xinyu_rtm/configuration_script.py

run_example5:
	./run_mlo.sh -f examples/reconfigurable_radio_new/fitness_script.py -c examples/reconfigurable_radio_new/configuration_script.py
    
run_example6:
	./run_mlo.sh -f examples/pq/fitness_script.py -c examples/pq/configuration_script.py
    
run_example_empty:
	./run_mlo.sh -f examples/empty/fitness_script.py -c examples/empty/configuration_script.py
    
run_trets_radio: $(SCRIPTS)
    
run_trets_quads_3d: $(SCRIPTS3D)
    
run_trets_quads_3d001: $(SCRIPTS3D001)
    
run_trets_quads_2d: $(SCRIPTS2D)

run_trets_quads_2d001: $(SCRIPTS2D001)

configuration_3d%:
	./run_mlo.sh -f examples/quadrature_method_based_app/fitness_script.py -c examples/trest_experiments/configuration_3d$*.py &> /tmp/mk3060.013d$*.txt
    
configuration_2d%:
	./run_mlo.sh -f examples/quadrature_method_based_app/fitness_script.py -c examples/trest_experiments/configuration_2d$*.py &> /tmp/mk30600012d_$*.txt
    
install_gui:
	$(EASY_INSTALL) numpy
	$(EASY_INSTALL) scipy
	$(EASY_INSTALL) scikit-learn
	$(EASY_INSTALL) matplotlib
	$(EASY_INSTALL) pisa
	$(EASY_INSTALL) pypdf
	$(EASY_INSTALL) wxPython pyt
	$(EASY_INSTALL) deap
	$(EASY_INSTALL) html5lib
	$(EASY_INSTALL) reportlab

    
install_terminal: 
	#yum install lapack lapack-devel blas blas-devel
	#$(EASY_INSTALL) numpy
	#$(EASY_INSTALL) scipy
	$(EASY_INSTALL) scikit-learn
	$(EASY_INSTALL) matplotlib
	$(EASY_INSTALL) pisa
	$(EASY_INSTALL) pypdf
	$(EASY_INSTALL) deap
	$(EASY_INSTALL) html5lib
	$(EASY_INSTALL) reportlab
	$(EASY_INSTALL) GitPython
	$(EASY_INSTALL) logging
	$(EASY_INSTALL) pyGtk
	$(EASY_INSTALL) argparse

    

restart:
	./run_mlo.sh --restart
    
pep:
	pep8 --statistic --exclude=pyXGPR,Old_Files .

.PHONY: presentation
presentation:
	pdflatex -output-directory=Presentation Presentation/presentation.tex
	pdflatex -output-directory=Presentation Presentation/presentation.tex

clean:
	rm profile
	find . -name "*.pyc" -delete
	find . -name "*~" -delete
    



