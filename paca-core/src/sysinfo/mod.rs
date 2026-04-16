use std::path::Path;

pub fn available_disk_space(path: &Path) -> Result<u64, std::io::Error> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(stat.f_bavail as u64 * stat.f_frsize as u64)
    }
}

pub fn check_disk_space(path: &Path, needed: u64) -> Result<(), crate::error::PacaError> {
    match available_disk_space(path) {
        Ok(available) if available < needed => {
            Err(crate::error::PacaError::InsufficientDiskSpace { needed, available })
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_disk_space_returns_nonzero_for_valid_path() {
        let space = available_disk_space(Path::new("/")).unwrap();
        assert!(space > 0);
    }

    #[test]
    fn available_disk_space_returns_error_for_nonexistent_path() {
        let result = available_disk_space(Path::new("/nonexistent/path/that/does/not/exist"));
        assert!(result.is_err());
    }

    #[test]
    fn check_disk_space_returns_ok_when_sufficient() {
        check_disk_space(Path::new("/"), 1).unwrap();
    }

    #[test]
    fn check_disk_space_returns_error_when_insufficient() {
        let result = check_disk_space(Path::new("/"), u64::MAX);
        match result {
            Err(crate::error::PacaError::InsufficientDiskSpace { needed, .. }) => {
                assert_eq!(needed, u64::MAX);
            }
            other => panic!("expected InsufficientDiskSpace, got {:?}", other),
        }
    }
}
