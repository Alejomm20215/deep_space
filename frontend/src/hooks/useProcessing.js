import { useState, useEffect, useRef, useCallback } from 'react';

export const useProcessing = () => {
    const [status, setStatus] = useState('idle'); // idle, uploading, processing, complete, error
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState('');
    const [detail, setDetail] = useState('');
    const [result, setResult] = useState(null);
    const [jobId, setJobId] = useState(null);

    const wsRef = useRef(null);

    const connectWebSocket = useCallback((id) => {
        // In production, use wss:// and proper host
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${id}`;
        // If running with proxy, the above works. If direct:
        // const wsUrl = `ws://localhost:8000/ws/${id}`;

        wsRef.current = new WebSocket(wsUrl);

        wsRef.current.onopen = () => {
            console.log('Connected to processing stream');
        };

        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleMessage(data);
        };

        wsRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            // setStatus('error'); 
        };

        wsRef.current.onclose = () => {
            console.log('Connection closed');
        };
    }, []);

    const handleMessage = (data) => {
        switch (data.type) {
            case 'init':
                updateStateFromStatus(data.status);
                break;
            case 'progress':
                setStatus('processing');
                setStage(data.stage);
                setProgress(data.progress);
                setDetail(data.detail);
                break;
            case 'complete':
                setStatus('complete');
                setProgress(100);
                setResult(data.result);
                break;
            case 'error':
                setStatus('error');
                setDetail(data.message);
                break;
            default:
                break;
        }
    };

    const updateStateFromStatus = (jobStatus) => {
        setStatus(jobStatus.status);
        setProgress(jobStatus.progress);
        setStage(jobStatus.stage);
        if (jobStatus.result) {
            setResult(jobStatus.result);
        }
    };

    const startUpload = async (files, mode) => {
        setStatus('uploading');
        setProgress(0);

        const formData = new FormData();

        // Handle FileList or single File or Array
        if (files instanceof FileList || Array.isArray(files)) {
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }
        } else {
            formData.append('files', files);
        }

        formData.append('mode', mode);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            setJobId(data.job_id);
            connectWebSocket(data.job_id);

        } catch (err) {
            setStatus('error');
            setDetail(err.message);
        }
    };

    const reset = () => {
        if (wsRef.current) wsRef.current.close();
        setStatus('idle');
        setProgress(0);
        setStage('');
        setDetail('');
        setResult(null);
        setJobId(null);
    };

    return {
        status,
        progress,
        stage,
        detail,
        result,
        startUpload,
        reset
    };
};
