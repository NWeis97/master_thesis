# SETTINGS 
run_stats_for='Our' # Choose between 'Our','Weis','Peter','Carl','Patrick','Rasmus'
speed=1 # Determines speed of run



########################
# SCRIPT (do not touch!)
########################
echo "Hello! Please hold on till setup is complete."

wait 2

if [ -d "/test_env" ] 
then
    echo ""
    echo ""
    echo "---------------------------------------------------------------"
    echo "Making tmp virtual enviroment and installing needed packages..."
    echo "---------------------------------------------------------------"
    wait 3


    # Make virtual python env
    python3 -m venv test_env

    # Source virtual enviroment
    source test_env/bin/activate

    # Download packages
    python3 -m pip install numpy
    python3 -m pip install pandas
    python3 -m pip install termplotlib
    python3 -m pip install pyfiglet
    python3 -m pip install XlsxWriter

    echo ""
    echo ""
    echo "---------------------------------------"
    echo "Virtual enviroment has now been set up!"
    echo "---------------------------------------"
    wait 3
else
    echo "Virtual enviroment already exists! Using existing enviroment"
    
    # Source virtual enviroment
    source test_env/bin/activate



echo "-------------------------"
echo "Ready to run application!"
echo "-------------------------"
wait 2


# Run python script
python3 SI_enjoyers_beer.py ${run_stats_for} ${speed}


# Deactivate virtual enviroment
echo "Deactivating virtual enviroment..."

deactivate 

echo "Done !"

echo " "
echo " "
echo " "



