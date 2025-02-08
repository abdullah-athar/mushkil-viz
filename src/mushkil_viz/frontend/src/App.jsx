import React, { useState } from 'react';
import {
    Container,
    Title,
    Text,
    Alert,
    Button,
    Group,
    ActionIcon,
    Paper,
    Stack,
    Box
} from '@mantine/core';
import { Notifications, showNotification } from '@mantine/notifications';
import { IconAlertCircle, IconSun, IconMoon, IconArrowBack } from '@tabler/icons-react';
import DropZone from './components/DropZone';
import VisualizationDashboard from './components/VisualizationDashboard';

// Get environment variables
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
const API_BASE_PATH = import.meta.env.VITE_API_BASE_PATH || '/api';

function App({ colorScheme, toggleColorScheme }) {
    const [visualizations, setVisualizations] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const isDark = colorScheme === 'dark';

    const handleDrop = async (file) => {
        setIsLoading(true);
        setError(null);
        setVisualizations(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            console.log('Uploading file:', file.name);
            const response = await fetch(`${BACKEND_URL}${API_BASE_PATH}/analyze`, {
                method: 'POST',
                body: formData,
            });

            const contentType = response.headers.get('content-type');
            let data;

            try {
                data = await response.json();
            } catch (e) {
                console.error('Failed to parse response:', e);
                throw new Error('Failed to parse server response');
            }

            if (!response.ok) {
                console.error('Server error response:', data);
                throw new Error(data.detail || 'Analysis failed');
            }

            console.log('Received data:', data);
            setVisualizations(data);

            showNotification({
                title: 'Success',
                message: 'Analysis completed successfully',
                color: 'green',
            });
        } catch (error) {
            console.error('Error:', error);
            setError(error.message || 'Failed to analyze the file');
            showNotification({
                title: 'Error',
                message: error.message || 'Failed to analyze the file',
                color: 'red',
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleStartNew = () => {
        setVisualizations(null);
        setError(null);
    };

    return (
        <Box bg={isDark ? 'dark.8' : 'gray.0'} style={{ minHeight: '100vh' }}>
            <Notifications />
            <Container size="xl" py="xl">
                {/* Header */}
                <Paper
                    shadow="sm"
                    p="md"
                    mb="xl"
                    style={{
                        position: 'sticky',
                        top: 0,
                        zIndex: 100,
                    }}
                    bg={isDark ? 'dark.7' : 'white'}
                >
                    <Group position="apart" align="center">
                        <Group>
                            {visualizations && (
                                <ActionIcon
                                    variant="light"
                                    onClick={handleStartNew}
                                    size="lg"
                                    color={isDark ? 'blue.4' : 'blue'}
                                    title="Start new analysis"
                                >
                                    <IconArrowBack size={20} />
                                </ActionIcon>
                            )}
                            <Title order={1} size="h2" c={isDark ? 'gray.1' : 'dark'}>
                                MushkilViz Data Analysis
                            </Title>
                        </Group>
                        <ActionIcon
                            variant="light"
                            onClick={() => toggleColorScheme()}
                            size="lg"
                            color={isDark ? 'yellow' : 'blue'}
                            title="Toggle color scheme"
                        >
                            {isDark ? <IconSun size={20} /> : <IconMoon size={20} />}
                        </ActionIcon>
                    </Group>
                </Paper>

                {/* Main Content */}
                <Stack spacing="xl">
                    {!visualizations ? (
                        <>
                            <Text size="lg" align="center" c={isDark ? 'dimmed' : 'dimmed'}>
                                Upload your CSV file to start analyzing your data
                            </Text>
                            <DropZone onDrop={handleDrop} isLoading={isLoading} isDark={isDark} />
                        </>
                    ) : (
                        <>
                            <Text size="sm" c={isDark ? 'dimmed' : 'dimmed'}>
                                Detected domain: {visualizations.domain || 'Unknown'}
                            </Text>
                            <VisualizationDashboard data={visualizations} colorScheme={colorScheme} />
                        </>
                    )}

                    {error && (
                        <Alert
                            icon={<IconAlertCircle size={16} />}
                            title="Error"
                            color="red"
                            variant="filled"
                        >
                            {error}
                        </Alert>
                    )}
                </Stack>
            </Container>
        </Box>
    );
}

export default App; 