from app.jobs import JobStore


def test_create_returns_queued_job_with_unique_id():
    store = JobStore(ttl_seconds=3600)
    job1 = store.create()
    job2 = store.create()

    assert job1.status == "queued"
    assert job1.result is None
    assert job1.error is None
    assert job1.job_id != job2.job_id


def test_get_returns_created_job():
    store = JobStore(ttl_seconds=3600)
    job = store.create()

    assert store.get(job.job_id) is job


def test_get_returns_none_for_unknown_id():
    store = JobStore(ttl_seconds=3600)

    assert store.get("does-not-exist") is None


def test_mark_completed_sets_status_and_result():
    store = JobStore(ttl_seconds=3600)
    job = store.create()

    store.mark_completed(job.job_id, {"text": "hello"})

    fetched = store.get(job.job_id)
    assert fetched.status == "completed"
    assert fetched.result == {"text": "hello"}
    assert fetched.error is None


def test_mark_failed_sets_status_and_error():
    store = JobStore(ttl_seconds=3600)
    job = store.create()

    store.mark_failed(job.job_id, {"message": "boom", "type": "server_error"})

    fetched = store.get(job.job_id)
    assert fetched.status == "failed"
    assert fetched.error == {"message": "boom", "type": "server_error"}
    assert fetched.result is None


def test_purge_expired_removes_old_jobs_using_injected_clock():
    fake_time = [0.0]
    store = JobStore(ttl_seconds=10, clock=lambda: fake_time[0])

    old_job = store.create()
    fake_time[0] = 20.0  # 20s later, past ttl_seconds=10
    new_job = store.create()  # create() purges expired jobs first

    assert store.get(old_job.job_id) is None
    assert store.get(new_job.job_id) is new_job
