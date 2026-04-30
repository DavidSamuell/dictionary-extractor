# Start Label Studio with local file serving enabled
# Then run the setup script to create/update projects

LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)/.label-studio-renders" \
LABEL_STUDIO_HOST=https://sudie-unfelted-neriah.ngrok-free.dev \
label-studio --enable-legacy-api-token &

# Wait for Label Studio to finish starting up
sleep 5

python label-studio/setup.py \
    --samples-dir assets/dictionaries/samples-2 \
    --ls-token cce752c0343c343e6e1958ef12117fb466581a4a \
    --render-dir .label-studio-renders
