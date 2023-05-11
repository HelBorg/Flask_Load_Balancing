start code 
mpiexec -np 1 python appfc\__init__.py
mpiexec -np 1 flask --app appfc\__init__.py run