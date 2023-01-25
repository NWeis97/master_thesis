# SETTINGS
run_stats_for='Our' # Choose between 'Our','Weis','Peter','Carl','Patrick','Rasmus'
speed=1 # Determines speed of run
gnu_plot='False' # Set to 'False' if gnuplot has not been installed on computer



########################
# SCRIPT (do not touch!)
########################
echo ""
echo ""
echo "---------------------------------------------"
echo "Hello! Please hold on till the setup is complete."
echo "---------------------------------------------"

sleep 1

if [ -d "./test_env/" ]
then
  echo "Virtual environment already exists! Using existing environment"

  # Source virtual enviroment
  source ./test_env/bin/activate
else
    echo ""
    echo ""
    echo "---------------------------------------------------------------"
    echo "Making tmp virtual environment and installing needed packages..."
    echo "---------------------------------------------------------------"
    sleep 1


    # Make virtual python env
    python3 -m venv test_env

    # Source virtual enviroment
    source ./test_env/bin/activate

    # Download packages
    python3 -m pip install numpy
    python3 -m pip install pandas
    python3 -m pip install termplotlib
    python3 -m pip install pyfiglet
    python3 -m pip install XlsxWriter
    python3 -m pip install matplotlib
    python3 -m pip install plotext

    echo ""
    echo ""
    echo "---------------------------------------"
    echo "Virtual environment has now been set up!"
    echo "---------------------------------------"
    sleep 1
fi


echo ""
echo ""
echo "-------------------------"
echo "Starting application...!"
echo "-------------------------"


# Run python script
if [ "$gnu_plot" == "False" ]
then
  python3 SI_enjoyers_beer_with_plotext.py ${run_stats_for} ${speed}
else
  python3 SI_enjoyers_beer.py ${run_stats_for} ${speed}
fi


# Deactivate virtual enviroment
echo "Deactivating virtual environment..."

deactivate

while true; do
    read -p "Do you wish to uninstall the virtual enviroment? (y/n): " yn
    case $yn in
        [Yy]* ) rm -rf ./test_env; echo "Virtual environment successfully uninstalled"; break;;
        [Nn]* ) break;;
        * ) echo "Please answer y or n";;
    esac
done

echo "Done !"

echo " "
echo " "
echo " "
