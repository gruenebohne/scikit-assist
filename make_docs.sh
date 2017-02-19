
rm -R docs/*
cd skassist-docs
make html
cd ..
rm -R docs/doctrees
cp -R docs/html/* docs/
rm -R docs/html
touch docs/.nojekyll