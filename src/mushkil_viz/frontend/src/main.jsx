import React from 'react';
import { createRoot } from 'react-dom/client';
import { MantineProvider } from '@mantine/core';
import { useColorScheme } from '@mantine/hooks';
import App from './App';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';

function Root() {
    const preferredColorScheme = useColorScheme();
    const [colorScheme, setColorScheme] = React.useState(preferredColorScheme);
    const toggleColorScheme = (value) => {
        const nextColorScheme = value || (colorScheme === 'dark' ? 'light' : 'dark');
        setColorScheme(nextColorScheme);
        // You might want to store this in localStorage for persistence
        try {
            window.localStorage.setItem('mantine-color-scheme', nextColorScheme);
        } catch (e) {
            console.warn('Failed to save color scheme to localStorage:', e);
        }
    };

    // Initialize from localStorage if available
    React.useEffect(() => {
        try {
            const savedScheme = window.localStorage.getItem('mantine-color-scheme');
            if (savedScheme) {
                setColorScheme(savedScheme);
            }
        } catch (e) {
            console.warn('Failed to read color scheme from localStorage:', e);
        }
    }, []);

    return (
        <MantineProvider
            theme={{
                colorScheme,
                primaryColor: 'blue',
                defaultRadius: 'md',
            }}
        >
            <App colorScheme={colorScheme} toggleColorScheme={toggleColorScheme} />
        </MantineProvider>
    );
}

const container = document.getElementById('root');
const root = createRoot(container);
root.render(
    <React.StrictMode>
        <Root />
    </React.StrictMode>
); 