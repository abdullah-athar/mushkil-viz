import React from 'react';
import { Paper, Text, LoadingOverlay } from '@mantine/core';
import { useDropzone } from 'react-dropzone';

function DropZone({ onDrop, isLoading, isDark }) {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop: files => onDrop(files[0]),
        accept: {
            'text/csv': ['.csv']
        },
        multiple: false
    });

    return (
        <Paper
            p="xl"
            mt="md"
            bg={isDark ? 'dark.6' : 'white'}
            style={{
                border: `2px dashed ${isDark ? '#5c5f66' : '#ced4da'}`,
                borderRadius: '8px',
                cursor: 'pointer',
                position: 'relative',
                minHeight: '200px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 200ms ease',
                '&:hover': {
                    borderColor: isDark ? '#909296' : '#228be6'
                }
            }}
            {...getRootProps()}
        >
            <LoadingOverlay visible={isLoading} />
            <input {...getInputProps()} />
            <div style={{ textAlign: 'center' }}>
                {isDragActive ? (
                    <Text size="xl" c={isDark ? 'gray.3' : 'gray.7'}>
                        Drop the CSV file here...
                    </Text>
                ) : (
                    <Text size="xl" c={isDark ? 'gray.3' : 'gray.7'}>
                        Drag and drop a CSV file here, or click to select
                    </Text>
                )}
            </div>
        </Paper>
    );
}

export default DropZone; 