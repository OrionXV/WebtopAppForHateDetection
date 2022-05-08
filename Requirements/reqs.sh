virtualenv -q -p /usr/bin/python3.9 $1
source $1/bin/activate

#$1/bin/apt-get install wkhtmltopdf
$1/bin/pip install -r requirements.txt

sudo apt-get install wkhtmltopdf -y